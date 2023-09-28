# -*- coding: utf-8 -*-
"""# Generic functions"""

def noise(var, dims, device='cpu'):
    '''
    Function that returns a complex tensor whose elements follow a complex
    Gaussian of zero-mean and variance 'var'.
    Inputs:
        - var: variance of each component of the multivariate Gaussian. Real
        tensor.
        - dims: dimensions of the output noise. Tuple or list of integers.
        - device: 'cpu' or 'cuda'.
    Output:
        Tensor of shape 'dims', whose elements are distributed as CN(0,var).
    '''

    return torch.sqrt(var/2.0) * (torch.randn(dims, device=device) + 1j * torch.randn(dims, device=device))

def generateImpairments(std_d, lamb, K):
    '''
    Function that generates random inter-antenna spacings so that the antenna
    positions are likely to be ordered
    Inputs:
        - std_d: standard deviation of the random perturbation.
        - lamb: carrier wavelength.
        - K: number of antennas in the ULA.
    Output:
        - d_abs: tensor that contains the perturbed inter-antenna spacings.
        Shape: (K, 1)
    '''
    d_random = lamb/2*torch.ones(K,1) + std_d*torch.randn(K,1)
    d_sorted, _ = torch.sort(d_random, dim=0)
    d_abs = torch.cat((torch.flip(d_sorted[::2,0], dims=[0]), d_sorted[1::2,0]),dim=0).view(-1,1)
    #We should always check that the antenna positions are ordered afterwards

    return d_abs

def steeringMatrix(theta_min, theta_max, K, d, lamb, device='cpu'):
    '''
    Function that returns a matrix whose columns are steering vectors of the
    form e^(-j*2*pi/lamb*d*k*sin(theta)), where k=-(K-1)/2,...,(K-1)/2 and
    theta~U(theta_min, theta_max).
    Inputs
        - theta_min: minimum of the angle uncertainty region. Real tensor with
        batch_size elements.
        - theta_max: maximum of the angle uncertainty region. Real tensor with
        batch_size elements. It should have the same shape as theta_min.
        - K: number of antenna elements. Real number.
        - d: inter-antenna spacing. It can be a vector or a scalar. For a scalar
        the same spacing is applied to all the elements in the antenna array.
        The vector should be of shape (K,1). By default is half the wavelength
        - lamb: carrier wavelength. Real number.
        - device: 'cpu' or 'cuda'.
    Outputs:
        - steeringMatrix: matrix whose columns are steering vectors. Shape: (K,
        batch_size)
        - theta: realization of the random variable of the angle. Shape: (1,
        batch_size)
    '''
    theta = torch.rand(theta_min.shape, device=device) * (theta_max - theta_min) + theta_min #Uniform in [theta_min, theta_max)
    theta = theta.view(1, -1)   #Row vector to have steering vectors in columns at the output

    steeringMatrix = torch.exp(-1j * 2 * np.pi / lamb * d * \
                               torch.arange(-(K-1)/2.0,(K-1)/2.0+1, device=device).view(-1,1).type(torch.cfloat) \
                               @ torch.sin(theta.type(torch.cfloat)))
    return steeringMatrix, theta

def rhoMatrix(range_min, range_max, S, Delta_f, device='cpu'):
    '''
    Function that returns a matrix whose columns are phase shifts of the
    form e^(-j*2*pi*s*Delta_f*2*R/c), where s=0,...,S-1 and
    R~U(range_min, range_max).
    Inputs
        - range_min: minimum of the range uncertainty region. Real tensor with
        batch_size elements.
        - range_max: maximum of the range uncertainty region. Real tensor with
        batch_size elements. It should have the same shape as range_min.
        - S: number of subcarriers in an OFDM symbol. Real number.
        - Delta_f: subcarrier spacing. Real number.
        - device: 'cpu' or 'cuda'.
    Outputs:
        - rhoMatrix: matrix whose columns are range-dependent phase shifts.
        Shape: (S, batch_size)
        - range: realization of the random variable of the range. Shape: (1,
        batch_size)
    '''
    range_tgt = torch.rand(range_min.shape, device=device) * (range_max - range_min) + range_min #Uniform in [range_min, range_max)
    range_tgt = range_tgt.view(1, -1)   #Row vector to have steering vectors in columns at the output

    rhoMatrix = torch.exp(-1j * 2 * np.pi * Delta_f * 2/3e8 * \
                          torch.arange(S, device=device).view(-1,1).type(torch.cfloat) \
                          @ range_tgt.type(torch.cfloat))
    return rhoMatrix, range_tgt

def createInterval(mean_min, mean_max, span_min, span_max, batch_size=1, device='cpu'):
    '''
    Function that creates an interval [minimum, maximum] as
    [minimum, maximum] = mean + [span/2, -span/2], where
    mean~U(mean_min, mean_max), and span~U(span_min, span_max).
    Inputs:
        - mean_min: minimum value of mean. Real number.
        - mean_max: maximum value of mean. Real number.
        - span min: minimum value of span. Real number.
        - span_max: maximum value of span. Real number.
        - batch_size: number of intervals we want to randomly draw. Integer.
        - device: 'cpu' or 'cuda'.
    Outputs:
        - minimum: minimum value of the intervals. Shape: (batch_size,1).
        - maximum: maximum value of the intervals. Shape: (batch_size,1).
    '''
    mean = torch.rand(batch_size, 1, device=device) * (mean_max - mean_min) + mean_min
    span = torch.rand(batch_size, 1, device=device) * (span_max - span_min) + span_min
    minimum = mean - span / 2.0
    maximum = mean + span / 2.0

    return minimum, maximum


def createPrecoder(theta_mean_min, theta_mean_max, span_min, span_max, K, batch_size, d, lamb, Ngrid = 256, device='cpu'):
    '''
    Function that creates a precoder based on a LS solution
    x_out = min_x ||b-A^Tx||^2. Considering Ngrid angles between [-pi/2, pi/2],
    b is a vector such that [b]_i=K if theta_i is in [theta_min, theta_max] and
    0 otherwise. The matrix A has columns that consists of steering vectors for
    each angle of the grid.
    [theta_min, theta_max] is randomly distributed. It is computed as
    [theta_min, theta_max] = theta_mean + [theta_span/2, -theta_span/2], where
    theta_mean~U(theta_mean_min, theta_mean_max) and theta_span~U(span_min,
    span_max).
    Input:
        - theta_mean_min: minimum value of the mean angle. Real number.
        - theta_mean_max: maximum value of the mean angle. Real number.
        - span_min: minimum value of the angle span. Real number.
        - span_max: maximum value of the angle span. Real number.
        - K: number of antenna elements of the transmitter.
        - batch_size: size of the batch to process. Integer.
        - d: inter-antenna spacing. By default is half the wavelength. Real
        number.
        - lamb: carrier wavelength. Real number.
        - Ngrid: number of grid angles to consider between [-pi/2, pi/2]
        - device: 'cpu' or 'cuda'
    Output:
        - precoder: matrix whose columns are precoding vectors for different
        batch samples. Shape: (K, batch_size)
        - theta_min: drawn instances of the minimum of the target angular
        sector. Shape: (batch_size, 1)
        - theta_max: drawn instances of the maximum of the target angular
        sector. Shape: (batch_size, 1)
        - A: matrix whose columns are steering vectors for an oversampled grid
        of angles. Shape: (K, Ngrid).
        - b_matrix: binary matrix whose columns are 1 for angles within
        [theta_min, theta_max], and 0 otherwise. Shape: (Ngrid, batch_size)
    '''
    angle_vector = torch.linspace(-np.pi/2, np.pi/2, Ngrid, device = device)
    # Create matrix (Ngrid_angle x b_size) whose columns correspond to 1 for angles within angular sector
    theta_min, theta_max = createInterval(theta_mean_min, theta_mean_max, span_min, span_max, batch_size, device=device)
    b_matrix = K*((angle_vector >= theta_min) & (angle_vector <= theta_max)).type(torch.cfloat).transpose(1,0)

    #Create matrix A, shape (K, Ngrid)
    A,_ = steeringMatrix(angle_vector, angle_vector, K, d, lamb, device=device)

    #By using a matrix B instead of a vector b, we can compute batch_size precoders at once
    precoder = torch.linalg.lstsq(A.transpose(1,0),b_matrix).solution
    #Apply a normalization layer so that each column has unit norm
    dividing_factor = torch.sum(torch.abs(precoder)**2,dim=0, keepdim=True)
    precoder = precoder / torch.sqrt(dividing_factor)

    return precoder, theta_min, theta_max, A, b_matrix/K

"""### Sensing functions"""

def sensingChannel(sigma_r, theta_min, theta_max, precoder, symbols, range_min, \
                   range_max, N0, presence, Delta_f, lamb, d, device='cpu'):
    '''
    Function that computes the output of the sensing channel.
    Inputs:
        - sigma_r: standard deviation (std) of the complex channel gains. Real
        number.
        - theta_min: minimum of the angle uncertainty region. Real tensor with
        1 or batch_size elements.
        - theta_max: maximum of the angle uncertainty region. Real tensor with
        1 or batch_size elements. It should have the same shape as theta_min.
        - precoder: vector that steers the antenna energy into a particular
        directon (f in the sensing equation). Complex tensor of shape
        (batch_size, K,1) or (K,1) if the same precoder is used across samples.
        - symbols: communication symbols to transmit. Complex tensor of shape
        (batch_size, S, 1).
        - range_min: minimum of the range uncertainty region. Real tensor with
        1 or batch_size elements.
        - range_max: maximum of the range uncertainty region. Real tensor with
        1 or batch_size elements. It should have the same shape as range_min.
        - N0: variance of the complex Gaussian noise to be added at the receiver
        side. Real number.
        - presence: binary vector that indicates whether the target is present
        for each batch sample. Binary tensor of size (batch_size, 1, 1).
        - Delta_f: spacing between subcarriers [Hz]. Real number.
        - lamb: carrier wavelength. Real number.
        - d: spacing between antenna elements in the transmitter ULA. By default
        it is half the wavelength. Real number.
        - device: 'cpu' or 'cuda'.
    Outputs:
        - Yr: received signal at the sensing receiver. Shape: (batch_size, K, S)
        - true_theta: true angle of each potential target. Shape: (batch_size,1)
        - true_range: true range of each potential range. Shape: (batch_size,1)
    '''

    #Get batch size, antennas and subcarriers from input data
    batch_size = torch.numel(presence)
    K = precoder.shape[-2]
    S = symbols.shape[-2]

    #If angles are scalar values, repeat them to match the batch_size
    if torch.numel(theta_min) == 1:
        theta_min = theta_min * torch.ones((batch_size, 1), device=device)
    if torch.numel(theta_max) == 1:
        theta_max = theta_max * torch.ones((batch_size, 1), device=device)

    #Draw realizatons of random variables and compute phase shift vectors
    complex_gain = noise(sigma_r**2, (batch_size, 1, 1), device)

    steering_vector, true_theta = steeringMatrix(theta_min, theta_max, K, d, lamb, device)
    steering_vector = steering_vector.transpose(1,0).reshape((-1, K, 1))       #Use -1 instead of batch_size since theta_min could be scalar

    rho_vector, true_range = rhoMatrix(range_min, range_max, S, Delta_f, device)
    rho_vector = rho_vector.transpose(1,0).reshape((-1, S, 1))

    #Compute noiseless reflection from target. Shape: (batch_size, K, S)
    reflection = 1/np.sqrt(S) * complex_gain * steering_vector @ steering_vector.permute(0,2,1) \
                    @ precoder @ (symbols * rho_vector).permute(0,2,1)

    #Generate random noise
    rx_noise = noise(N0, (batch_size, K, S), device=device)

    #Include target presence and noise
    Yr = reflection*presence + rx_noise

    return Yr, true_theta.view(-1,1), true_range.view(-1,1)

"""### Communication functions"""

def createMessages(card, num=1, device='cpu'):
    '''
    Function that randomly creates a list of messages
    Input:
        - card: cardinality of the set of messages. Integer.
        - num: number of messages to transmit. Integer greater than 0.
        - device: 'cpu' or 'cuda'
    Output:
        - List of messages. List of integers with 'num' elements
    '''
    return torch.randint(0,card,size=(num,), dtype=torch.long, device=device)

def MPSK(M, rotation=0, device='cpu'):
    '''Function that returns an array with all the symbols of a M-PSK
    constellation.
    Inputs:
        - M: size of the constellation
        - rotation: angle of rotation of the constellation [rad]
    Output:
        - A list with all the M possible symbols of the constellation
    '''
    return torch.exp(1j * (2*np.pi/M * torch.arange(0,M, device=device) + rotation))

def commChannel(sigma_vector, theta_min, theta_max, K, d, lamb, precoder, x, N0, device='cpu'):
    '''Function that computes the received signal in a OFDM MISO communication
    channel.
    Inputs:
        - sigma_vector: vector of standard deviations per subcarrier.
        Shape: (1,S)
        - theta_min: minimum angle where the Comm. Rx can lie. Scalar value or
        tensor with batch_size elements.
        - theta_max: maximum angle where the Comm. Rx can lie. Scalar value or
        tensor with batch_size elements
        - K: number of antenna elements. Integer
        - precoder: ISAC precoder. Size (batch_size, K, 1)
        - x: transmitted complex symbols per subcarrier. Size: (batch_size, S, 1)
        - N0: variance of the noise. Scalar

    Outputs:
        - y_c: received signal. Shape: (batch_size, 1, S)
        - true_theta: true angle vector. Shape: (1, batch_size)
        - beta_vector_fft: beta vector applied DFT. Shape: (batch_size, S, 1)
    '''
    #Infere batch size and subcarriers from input data
    batch_size = precoder.shape[0]
    S = sigma_vector.shape[1]

    #If angles are scalar values, repeat them to match the batch_size
    if torch.numel(theta_min) == 1:
        theta_min = theta_min * torch.ones((batch_size, 1), device=device)
    if torch.numel(theta_max) == 1:
        theta_max = theta_max * torch.ones((batch_size, 1), device=device)

    #Define steering vectors and reshape them
    steering_vector, true_theta = steeringMatrix(theta_min, theta_max, K, d, lamb, device)
    steering_vector = steering_vector.T.reshape((-1, K, 1))

    #Define random taps in the delay domain
    beta_vector = sigma_vector/np.sqrt(2.0) * (torch.randn(batch_size, S, device=device) + 1j * torch.randn(batch_size, S, device=device))
    #Perform DFT along rows (along different taps)
    beta_vector_fft = torch.fft.fft(beta_vector, norm='ortho')
    #Reshape beta_vector (reading by rows)
    beta_vector_fft = beta_vector_fft.reshape((batch_size, S, 1))

    #Generate random noise
    n = noise(N0, (batch_size, 1, S), device)

    #Compute received signal
    y_c = steering_vector.permute(0,2,1) @ precoder @ (beta_vector_fft * x).permute(0,2,1) + n

    return y_c, true_theta, beta_vector_fft

def createKappa(b_vector, true_theta, K, d, lamb, precoder, device = 'cpu'):
    ''' Function that computes the CSI for the Rx.
    Inputs:
        - b_vector: DFT of the channel taps vector. Shape: (batch_size, S, 1)
        - true_theta: true value of the AoA at the comm. receiver.
        Shape: (batch_size, )
        - precoder: precoder used to transmit the signal.
        Shape: (batch_size, K, 1)
    Output:
        - kappa: complex value of CSI per batch. Shape: (batch_size, 1, S)
    '''

    #Infer batch size and number of antennas from input data
    batch_size = b_vector.shape[0]

    #Create steering matrix of size (batch_size, 1, K) from true angles
    # steering_vector = torch.exp(-1j * 2 * np.pi/lamb * d * torch.arange(-(K-1)/2.0,(K-1)/2.0+1, device=device).view(-1,1).type(torch.cfloat) @
    #                    torch.sin(true_theta.reshape(batch_size,1,1).type(torch.cfloat)))
    steering_vector, _ = steeringMatrix(true_theta, true_theta, K, d, lamb, device)
    steering_vector = steering_vector.transpose(-2,-1).view(batch_size, K, 1)

    #Compute CSI
    kappa = steering_vector.permute(0,2,1) @ precoder @ b_vector.permute(0,2,1)

    return kappa

def MLdecoder(rec, kappa, refConst):
    ''' Function that performs ML decoding per subcarrier in an OFDM system
    given some channel state information (CSI)
    Inputs:
        - rec: received signal at the Rx. Shape: (batch_size, 1, S)
        - kappa: channel state information. Shape: (batch_size, 1, S)
        - refConst: reference constellation to discern which message was
        transmitted. Shape: (numMessages,)
    Outputs:
        - est_message: estimated messages. Length: batch_size*S
    '''

    # Create quantity to compare with received signal
    compare = kappa * refConst.reshape(len(refConst),1)
    #This way is faster than temp * (refConst.reshape(len(refConst),1) @ b.permute(0,2,1))

    #Compare received signal with previous quantity
    metric = torch.abs(rec - compare)**2

    #Take the argmin to retrieve the message that minimizes the metric
    est_message = torch.argmin(metric, dim=1)

    return est_message.flatten()

"""### ISAC functions

"""

def createCommonPrecoder(f_radar, f_comm, rho, phi, batch_size):
    '''Function that creates a precoder for JRC using the technique of
    [A. Zhang et all]. We suppose that the precoder is unit-energy.
    Inputs:
        f_radar: radar precoder. Shape: (1, K)
        f_comm: communication precoder. Shape: (1, K)
        rho: trade-off parameter. Real tensor
        phi: trade-off parameter. Real tensor
        batch_size: number of samples of the batch. Real number
    '''

    #The precoder are supposed to be row vectors
    temp = torch.sqrt(rho) * f_radar + torch.sqrt(1-rho) * torch.exp(1j * phi) * f_comm
    temp = temp / torch.real(torch.sqrt(temp @ torch.conj(temp.T)))
    precoder = torch.repeat_interleave(temp, batch_size,dim=0)

    return precoder

"""## Training function"""

def trainNetwork(network, init_train_it, end_train_it, sigma_sens, theta_mean_min, theta_mean_max, theta_span_min,
                 theta_span_max, range_mean_min, range_mean_max, range_span_min,
                 range_span_max, numAntennas, numSubcarriers, Delta_f, lamb, Ngrid_angle,
                 range_grid, pixels_angle, pixels_range,
                 msg_card, refConst, N0, true_spacing,
                 batch_size, optimizer, scheduler, criterionLoss, epoch_test_list, theta_min_sens_test,
                 theta_max_sens_test, range_min_sens_test, range_max_sens_test,
                 target_pfa, delta_pfa, thresholds_pfa, nTestSamples, SL_flag = True, sch_flag=False, device='cpu'):
    '''
    Function that performs training of a network and returns the loss function
    across training iterations and intermediate evaluation results if desires.
    Inputs:
        - network: network to optimize. Instance of a class.
        - init_train_it: Epoch where to start training. Integer.
        - end_train_it: epoch where to stop training. Integer
        - sigma_sens: standard deviation of the channel complex gain. Float.
        - theta_mean_min: minimum value of the mean of the angular sector
        [rad]. Float.
        - theta_mean_max: maximum value of the mean of the angular sector
        [rad]. Float.
        - theta_span_min: minimum value of the span of the angular sector
        [rad]. Float.
        - theta_span_max: maximum value of the span of the angular sector
        [rad]. Float.
        - range_mean_min: minimum value of the mean of the range sector [m].
        Float.
        - range_mean_max: maximum value of the mean of the range sector [m].
        Float.
        - range_span_min: minimum value of the span of the range sector [m].
        Float.
        - range_span_max: maximum value of the span of the range sector [m].
        Float.
        - numAntennas: number of antenna elements in the ULA. Integer.
        - numSubcarriers: number of subcarriers of the OFDM signal. Integer.
        - Delta_f: spacing between different subcarriers [Hz]. Float.
        - lamb: wavelength [m]. Float.
        - Ngrid_angle: number of points to perform angle grid search. Integer.
        - range_grid: grid of ranges to perform range grid search. Float tensor.
        - pixels_angle: number of elements to look around the maximum of the
        angle-delay map in the angular dimension (see sensingReceiver).
        Integer.
        - pixels_range: number of elements to look around the maximum of the
        angle-delay map in the range dimension (see sensingReceiver).
        Integer.
        - refConst: reference communicaiton constellation. Complex tensor.
        - N0: variance of the receiver AWGN. Float.
        - true_spacing: true inter-antenna spacing. Float tensor of length
        numAntennas.
        - batch_size: size of the batch of samples. Integer.
        - optimizer: optimizer to use during backpropagation.
        - scheduler: scheduler to use during backpropagation.
        - criterionLoss: loss function to optimize the network.
        - epoch_test_list: list of epochs to test the network. List of integers.
        - theta_min_sens_test: minimum angle of the testing angular sector
        [rad]. Float.
        - theta_max_sens_test: maximum angle of the testing angular sector
        [rad]. Float.
        - range_min_sens_test: minimum angle of the testing range sector [m].
        Float.
        - range_max_sens_test: maximum angle of the testing range sector [m].
        Float.
        - target_pfa: target false alarm probability to test. Float.
        - delta_pfa: maximum allowable deviation from target_pfa. Float.
        - thresholds_pfa: List of thresholds to try to obtain target_pfa. Float
        tensor.
        - nTestSamples: number of test samples to use. Integer.
        - SL_flag: flag to indicate if supervised or unsupervised learning is
        applied. Default: 'True' (perform supervised learning)
        - sch_flag: flag to indicate whether scheduler is applied to the
        training process. By defaul it doesn't apply (False).
        - device: 'cpu' or 'cuda'. Default: 'cpu'.
    Output:
        - loss_np: loss across iterations. Float list of length
        end_train_it-init_train_it.
        - loss_val_np: position RMSE as validation loss. For SL this is the same
        as loss_np. Float list of length end_train_it-init_train_it.
        - pd_inter: probability of detection for the testing epochs. Float list
        of length len(epoch_test_list).
        - pfa_inter: false alarm probability for the testing epochs (just as
        a check). Float list of length len(epoch_test_list).
        - rmse_angle_inter: angle RMSE [rad] for the testing epochs. Float list
        of length len(epoch_test_list).
        - rmse_range_inter: range RMSE [m] for the testing epochs. Float list of
        length len(epoch_test_list).
        - rmse_pos_inter: position RMSE [m] for the testing epochs. Float list
        of length len(epoch_test_list).
    '''
    print('*** Training started ***')
    network.train()

    #List to save the loss function values and the tested iterations
    loss_np, loss_val_np, num_iterations = [], [], []

    #Lists to save intermediate results
    pd_inter, pfa_inter, rmse_angle_inter, rmse_range_inter, rmse_pos_inter = [], [], [], [], []

    for epoch in range(init_train_it, end_train_it):
        #For SL we always assume a target, but not for UL
        if SL_flag:
            target = torch.ones((batch_size,1,1), dtype=torch.float32, device=device)
        else:
            target = torch.randint(0,2,(batch_size,1,1), dtype=torch.float32, device=device)
        ### Sensing precoder ###
        precoder_sens, theta_min_sens, theta_max_sens, A_matrix, b_matrix_angle = \
                createPrecoder(theta_mean_min, theta_mean_max, theta_span_min,
                               theta_span_max, numAntennas, batch_size, d=network.d,
                               lamb=lamb, Ngrid=Ngrid_angle, device=device)  #Shape(K,batch_size)
        precoder_sens = precoder_sens.transpose(-2,-1).view((batch_size, numAntennas, 1))

        ### Generation of random variables ###
        #Generate range sectors from mean and span
        range_min_sens, range_max_sens = createInterval(range_mean_min, range_mean_max,
                                                        range_span_min, range_span_max,
                                                        batch_size, device=device)
        # Create binary matrix (b_size x 1 x N_grid_range) whose columns are 1 for ranges within the considered limits
        b_matrix_range = ((range_grid >= range_min_sens.view(batch_size,1,1)) \
                        & (range_grid <= range_max_sens.view(batch_size,1,1))).type(torch.cfloat)
        #Generate random messages to transmit
        msg_card = len(refConst)
        msg = createMessages(msg_card, num=batch_size*numSubcarriers, device=device)
        #Reshape msgs to the needs of the comm CH function
        symbols = refConst[msg].reshape(batch_size, numSubcarriers, 1)

        ### Sensing channel ###
        Y_r, true_angle, true_range = sensingChannel(sigma_sens, theta_min_sens, theta_max_sens, precoder_sens,
                                                    symbols, range_min_sens, range_max_sens, N0, target, Delta_f, lamb,
                                                    true_spacing, device=device)

        ### Sensing Receiver ###
        metric, est_angle, est_range = sensingReceiver(Y_r, symbols, A_matrix, b_matrix_angle,
                                                          b_matrix_range, range_grid, pixels_angle,
                                                          pixels_range, numSubcarriers, Delta_f,
                                                          d=network.d, lamb=lamb, device=device)

        ### Loss function and optimization step ###
        #For position RMSE, we only consider when there is a target
        target_bool = target.to(torch.bool).view(batch_size,)
        est_range = est_range[target_bool, :]
        est_angle = est_angle[target_bool, :]
        true_range = true_range[target_bool, :]
        true_angle = true_angle[target_bool, :]
        #Compute true and estimated positions from angle and range
        est_x = est_range * torch.cos(est_angle)
        est_y = est_range * torch.sin(est_angle)
        est_pos = torch.cat((est_x, est_y), dim=-1)

        true_x = true_range * torch.cos(true_angle)
        true_y = true_range * torch.sin(true_angle)
        true_pos = torch.cat((true_x, true_y), dim=-1)

        #Select loss function
        loss_super = criterionLoss(est_pos, true_pos)
        loss_val_np.append(loss_super.item())
        if SL_flag:
            loss = loss_super
        else:
            loss = torch.mean(metric[:,0])
        #Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Scheduler step
        if sch_flag:
            scheduler.step()

        ### Save loss function and print information ###
        loss_np.append(loss.item())

        if (epoch+1) in epoch_test_list:
            ### Test the network with the current number of iterations ###
            num_iterations.append(epoch+1)

            network.eval()
            pd_temp, pfa_temp, rmse_angle_temp, rmse_range_temp, rmse_pos_temp = \
                    testNetworkSensing(sigma_sens, theta_min_sens_test, theta_max_sens_test, range_min_sens_test,
                        range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                        numAntennas, numSubcarriers, N0, Delta_f, lamb, true_spacing, network.d, refConst,
                        target_pfa, delta_pfa, thresholds_pfa, batch_size, nTestSamples, device=device)

            pd_inter.append(pd_temp)
            pfa_inter.append(pfa_temp)
            rmse_angle_inter.append(rmse_angle_temp)
            rmse_range_inter.append(rmse_range_temp)
            rmse_pos_inter.append(rmse_pos_temp)
            network.train()

            print(f'{epoch+1 - init_train_it}/{end_train_it - init_train_it} completed training iterations.')
    return loss_np, loss_val_np, num_iterations, pd_inter, pfa_inter, rmse_angle_inter, rmse_range_inter, rmse_pos_inter

"""## Testing functions"""

def getSensingMetrics(est_presence, true_presence, threshold, est_angle=None,
                      true_angle=None, est_range=None, true_range=None, device='cpu'):
    '''
    Function that computes detection and false alarm probabilities, and angle,
    range and position RMSEs.
    Inputs:
        - est_presence: estimation of the presence of the target. Float list.
        - true_presence: true presence of the target. Binary list.
        - threshold: threshold to discern if there is a target. Float.
        - est_angle: estimation of the target angle. Float tensor.
        - true_angle: true target angle. Float tensor.
        - est_range: estimation of the target range. Float tensor.
        - true_range: true target range. Float tensor.
        - device: 'cpu' or 'cuda'. Default 'cpu'
    Output:
        - pd: detection probability. Float.
        - pfa: false alarm probability. Float.
        - angle_rmse: angle RMSE. Float.
        - range_rmse: range RMSE. Float.
        - pos_rmse: position RMSE. Float.
    '''
    #Distinguish whether there is a target
    binary_list = (est_presence > threshold) * 1.0
    #Probability of correct detection
    detection = ((binary_list == 1) & (true_presence == 1)).sum() * 1.0 / (true_presence == 1).sum()
    pd = detection.item()
    #Probability of false alarm
    false_positive = ((binary_list == 1) & (true_presence == 0)).sum() * 1.0 / (true_presence == 0).sum()
    pfa = false_positive.item()

    if (est_angle==None) or (est_range==None) or (true_angle==None) or (true_range == None):
        rmse_angle = rmse_range = rmse_pos = None
    else:
        #Compute position from angle and range
        nTestSamples = len(est_angle)
        true_position = torch.empty(nTestSamples, 2, dtype=torch.float32, device=device)
        true_position[:,0] = (true_range * torch.cos(true_angle)).flatten()
        true_position[:,1] = (true_range * torch.sin(true_angle)).flatten()
        est_position  = torch.empty(nTestSamples, 2, dtype=torch.float32, device=device)
        est_position[:,0] = (est_range * torch.cos(est_angle)).flatten()
        est_position[:,1] = (est_range * torch.sin(est_angle)).flatten()

        #RMSE computations
        true_presence_index = true_presence.to(torch.bool)
        max_list_index = binary_list.to(torch.bool)
        rmse_indexes = max_list_index & true_presence_index        #Boolean indexes where to compute RMSE
        rmse_angle = torch.sqrt(torch.mean((true_angle[rmse_indexes]
                                                -est_angle[rmse_indexes])**2)).item()
        rmse_range = torch.sqrt(torch.mean((true_range[rmse_indexes]
                                                -est_range[rmse_indexes])**2)).item()
        rmse_pos = torch.sqrt(torch.mean(torch.norm(true_position[rmse_indexes.flatten(),:]
                                                        - est_position[rmse_indexes.flatten(),:], dim=1)**2)).item()
    return pd, pfa, rmse_angle, rmse_range, rmse_pos

def obtainThresholdsFixedPfa(est_presence, true_presence, target_pfa, delta_pfa, init_thr, device='cpu'):
    '''
    Function that empirically estimates the thresholds to yield a target false
    alarm probability with a maximum allowable error.
    Since obtaining exactly pfa = target_pfa is very difficult, we compute 3
    pfa's, so that for any of those probabilities,
    target_pfa - delta_pfa < pfa < target_pfa + delta_pfa.
    We then linearly interpolate the results.
    Inputs:
        - est_presence: estimated presence of a target. Binary list whose
        length is nTestSamples.
        - true_presence: true presence of a target. Binary list whose length
        is nTestSamples.
        - target_pfa: target false alarm proability. Float.
        - delta_pfa: maximum allowable error for the target_pfa. Float.
        - init_thr: initial thresholds to start the algorithm. Float numpy array
        with more than 1 element (usually 3).
    Outputs
        - final_thr: final thresholds that achieve target_pfa - delta_pfa < pfa,
        pfa < target_pfa + delta_pfa.
    '''
    with torch.no_grad():
        #Reset Pfa
        pfa = np.zeros((1,len(init_thr)))   #Set initial Pfa to enter loop
        while ((np.max(pfa) > target_pfa + delta_pfa) or (np.min(pfa) < target_pfa - delta_pfa)):
            #Lists to save final results
            pd, pfa = [], []

            #Compute detection and false alarm probabilities, and MSEs
            for t in range(len(init_thr)):
                pd_temp, pfa_temp, _, _, _ =  getSensingMetrics(est_presence, true_presence, init_thr[t],
                                                                None, None, None, None, device)
                pd.append(pd_temp)
                pfa.append(pfa_temp)

            #Check if target_pfa - delta_pfa < pfa < target_pfa + delta_pfa. We use the std to update the threshold vector
            if target_pfa < np.min(pfa):
                init_thr += torch.std(init_thr)
            elif target_pfa > np.max(pfa):
                init_thr -= torch.std(init_thr)
            else:
                #Check that the pfa is not e.g [1,0,0]
                if np.max(pfa) > target_pfa + delta_pfa:
                    init_thr = torch.linspace((init_thr[0] + init_thr[1]) / 2.0, init_thr[2], 3, device=device)
                if np.min(pfa) < target_pfa - delta_pfa:
                    init_thr = torch.linspace(init_thr[0], (init_thr[1] + init_thr[2]) / 2.0, 3, device=device)

    return init_thr

def testNetworkSensing(sigma_sens, theta_min_sens_test, theta_max_sens_test, range_min_sens_test,
                       range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                       K, S, N0, Delta_f, lamb, true_d, network_d, refConst,
                       target_pfa, delta_pfa, thresholds, batch_size, nTestSamples, device='cpu'):
    '''
    Function to test the sensing performance of a network for a fixed false alarm probability.
    Inputs:
        - sigma_sens: standard deviation of the complex channel gain. non-
        negative Float.
        - theta_min_sens_test: minimum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - theta_max_sens_test: maximum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - range_min_sens_test: minimum range of the coarse range sector of the
        location of the target. non-negative Float.
        - range_max_sens_test: maximum range of the coarse range sector of the
        location of the target. non-negative Float.
        - Ngrid_angle: number of points to perform angle grid search
        - range_grid: grid of ranges to perform grid search for range estimation.
        Float tensor. The length of range_grid is denoted Ngrid_range.
        - pixels_angle: number of elements in the angle dimension to look around
        the max. of the angle-delay map. Integer.
        - pixels_range: number of elements in the range dimension to look around
        the max. of the angle-delay map. Integer.
        - K: number of antenna elements. Integer.
        - S: number of subcarriers. Integer.
        - N0: variance of the noise added at the receiver. non-negative Float.
        - Delta_f: spacing in Hz between different subcarriers. non-negative Float.
        - lamb: wavelength. non-negative Float.
        - true_d: true inter-antenna spacing to be used in the channel model.
        Float tensor of length K.
        - network_d: assumed inter-antenna spacing by the network. Float tensor
        of length K.
        - refConst: constellation of the transmitted communication symbols.
        Complex Tensor whose length is msg_card.
        - target_pfa: target false alarm probability (Pfa) to evaluate. Float.
        - delta_pfa: maximum allowed variation of the empirical Pfa with target_pfa.
        Float.
        - thresholds: thresholds to apply to distinguish if there is a target.
        Float tensor of length more than 1 (usually 3).
        - batch_size: size of the data batch. Integer.
        - nTestSamples: number of testing samples. Integer.
        - device: 'cuda' or 'cpu'. Default: 'cpu'.
    Outputs:
        - pd_inter: probability of detection. Float.
        - pfa_inter: probability of false alarm (to check with input). Float.
        - rmse_angle_inter: angle RMSE in rads. Float.
        - rmse_range_inter: range RMSE in meters. Float.
        - rmse_pos_inter: position RMSE in meters. Float.
    '''
    with torch.no_grad():
        #Get information from the input data
        msg_card = len(refConst)
        numTestIt = nTestSamples // batch_size
        #Create mean and span values from testing angular sector
        theta_mean_min_sens = theta_mean_max_sens = (theta_min_sens_test+theta_max_sens_test)/2.0
        span_min_theta_sens = span_max_theta_sens = theta_max_sens_test-theta_min_sens_test
        #Create matrix to compute the angle-delay map
        P_matrix, _ = rhoMatrix(range_grid, range_grid, S, Delta_f, device=device)

        #Create lists to save results in each iteration
        true_presence_list = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        true_angle_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        true_range_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_presence_list  = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_angle_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_range_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)

        #Create precoder for testing
        precoder_sens, _, _, A_matrix, b_matrix_angle = createPrecoder(theta_mean_min_sens, theta_mean_max_sens, span_min_theta_sens,
                                                                    span_max_theta_sens, K, batch_size, d=network_d, lamb=lamb,
                                                                    Ngrid=Ngrid_angle, device=device)
        precoder_sens = precoder_sens[:,0].view(-1,1)
        # Create binray matrix (b_size x 1 x N_grid_range) whose columns are 1 for ranges within the considered limits
        b_matrix_range = ((range_grid >= range_min_sens_test.view(1,1,1)) \
                        & (range_grid <= range_max_sens_test.view(1,1,1))).type(torch.cfloat)

        for i in range(numTestIt):
            #Generate random logic vector that tells if there is a target
            target = torch.randint(0,2,(batch_size,1,1), dtype=torch.float32, device=device)
            #Generate random messages to transmit
            msg = createMessages(msg_card, num=batch_size*S, device=device)
            #Reshape msgs to the needs of the comm CH function
            symbols = refConst[msg].reshape(batch_size, S, 1)

            #Sensing channel
            Y_r, true_angle, true_range = sensingChannel(sigma_sens, theta_min_sens_test, theta_max_sens_test,
                                                        precoder_sens, symbols, range_min_sens_test,
                                                        range_max_sens_test, N0, target, Delta_f, lamb,
                                                        true_d, device)

            ### Sensing Receiver ###
            max_admap, est_angle, est_range = sensingReceiver(Y_r, symbols, A_matrix, b_matrix_angle,
                                                                 b_matrix_range, range_grid, pixels_angle,
                                                                 pixels_range, S, Delta_f,
                                                                 d=network_d, lamb=lamb, device=device)

            #Save true values
            true_presence_list[i*batch_size:(i+1)*batch_size] = target.view(batch_size,1)
            true_angle_list[i*batch_size:(i+1)*batch_size] = true_angle
            true_range_list[i*batch_size:(i+1)*batch_size] = true_range
            #Save estimations
            est_presence_list[i*batch_size:(i+1)*batch_size] = max_admap.view(-1,1)
            est_angle_list[i*batch_size:(i+1)*batch_size] = est_angle
            est_range_list[i*batch_size:(i+1)*batch_size] = est_range

        #Get thresholds that give relatively close to the target Pfa
        init_thr = torch.clone(thresholds)      #To avoid that the next function overwrites the thresholds.
        final_thr = obtainThresholdsFixedPfa(est_presence_list, true_presence_list, target_pfa, delta_pfa, init_thr, device)

        #Lists to save final results
        pd, pfa, rmse_angle, rmse_range, rmse_pos = [], [], [], [], []

        #Compute detection and false alarm probabilities, and RMSEs
        for t in range(len(final_thr)):
            pd_temp, pfa_temp, rmse_angle_temp, rmse_range_temp, rmse_pos_temp = getSensingMetrics(est_presence_list, true_presence_list,
                                                                                                   final_thr[t], est_angle_list,
                                                                                                   true_angle_list, est_range_list,
                                                                                                   true_range_list, device)
            pd.append(pd_temp)
            pfa.append(pfa_temp)
            rmse_angle.append(rmse_angle_temp)
            rmse_range.append(rmse_range_temp)
            rmse_pos.append(rmse_pos_temp)

        #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa (sorting avoids problems with values very similar but not equal)
        pd_inter = np.interp(1e-2,np.sort(pfa),np.sort(pd))
        pfa_inter = np.array(pfa).mean()
        rmse_angle_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_angle))
        rmse_range_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_range))
        rmse_pos_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_pos))

    return pd_inter, pfa_inter, rmse_angle_inter, rmse_range_inter, rmse_pos_inter

def testBaselineSensingFixPfa(sigma_sens, theta_min_sens_test, theta_max_sens_test, \
                              range_min_sens_test, range_max_sens_test, K, S, N0, Delta_f, \
                              lamb, true_d, assumed_d, Ngrid_angle, Ngrid_range, \
                              refConst, target_pfa, delta_pfa, \
                              thresholds, batch_size, nTestSamples, \
                              device='cpu'):
    '''
    Function to test the sensing performance of the baseline for a fixed false
    alarm probability.
    Inputs:
        - sigma_sens: standard deviation of the complex channel gain. non-
        negative Float.
        - theta_min_sens_test: minimum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - theta_max_sens_test: maximum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - range_min_sens_test: minimum range of the coarse range sector of the
        location of the target. non-negative Float.
        - range_max_sens_test: maximum range of the coarse range sector of the
        location of the target. non-negative Float.
        - K: number of antenna elements. Integer.
        - S: number of subcarriers. Integer.
        - N0: variance of the noise added at the receiver. non-negative Float.
        - Delta_f: spacing in Hz between different subcarriers. non-negative Float.
        - lamb: wavelength. non-negative Float.
        - true_d: true inter-antenna spacing to be used in the channel model.
        Float tensor of length K.
        - assumed_d: assumed inter-antenna spacing. Float tensor of length K.
        - Ngrid_angle: number of grid points to test in the angular domain.
        - Ngrid_range: number of grid points to test in the range domain.
        - refConst: constellation of the transmitted communication symbols.
        Complex Tensor whose length is msg_card.
        - target_pfa: target false alarm probability (Pfa) to evaluate. Float.
        - delta_pfa: maximum allowed variation of the empirical Pfa with target_pfa.
        Float.
        - thresholds: thresholds to apply to distinguish if there is a target.
        Float tensor of length more than 1 (usually 3).
        - batch_size: size of the data batch. Integer.
        - nTestSamples: number of testing samples. Integer.
        - device: 'cuda' or 'cpu'. Default: 'cpu'.
    Outputs:
        - pd_inter: probability of detection. Float.
        - pfa_inter: probability of false alarm (to check with input). Float.
        - rmse_angle_inter: angle RMSE in rads. Float.
        - rmse_range_inter: range RMSE in meters. Float.
        - rmse_pos_inter: position RMSE in meters. Float.
    '''
    with torch.no_grad():
        #Infer data from inputs
        numTestIt = nTestSamples // batch_size
        msg_card = len(refConst)
        #Create mean and span angles from given angular sector
        theta_mean_min_sens = theta_mean_max_sens = (theta_min_sens_test+theta_max_sens_test)/2.0
        span_min_theta_sens = span_max_theta_sens = theta_max_sens_test-theta_min_sens_test

        #Create lists to save results in each iteration
        true_presence_list = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        true_angle_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        true_range_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_presence_list  = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_angle_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
        est_range_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)

        #Create precoder for testing (ideal spacing). batch_size=1 because angles are fixed
        precoder_sens, _, _, _, _ = createPrecoder(theta_mean_min_sens, theta_mean_max_sens, \
                                                span_min_theta_sens, span_max_theta_sens, K, \
                                                batch_size=1, d=assumed_d, lamb=lamb, \
                                                Ngrid = Ngrid_angle, device=device)

        for i in range(numTestIt):
            #Generate random logic vector that tells if there is a target
            t = torch.randint(0,2,(batch_size,1,1), dtype=torch.float32, device=device)
            #Generate random messages to transmit
            msg = createMessages(msg_card, num=batch_size*S, device=device)
            #Reshape msgs to the needs of the comm CH function
            symbols = refConst[msg].reshape(batch_size, S, 1)

            #Sensing channel (true antenna spacing)
            Y_r, true_angle, true_range = sensingChannel(sigma_sens, theta_min_sens_test, theta_max_sens_test, \
                                                        precoder_sens, symbols, range_min_sens_test, \
                                                        range_max_sens_test, N0, t, Delta_f, lamb, \
                                                        true_d, device)

            #Sensing receiver
            max_admap, est_angle, est_range = sensingReceiverBaseline(Y_r, symbols, theta_min_sens_test, \
                                                                    theta_max_sens_test, range_min_sens_test, \
                                                                    range_max_sens_test, Delta_f, \
                                                                    d=assumed_d, lamb=lamb,\
                                                                    numSamplesAngle = Ngrid_angle, \
                                                                    numSamplesRange = Ngrid_range, \
                                                                    device = device)

            #Save true values
            true_presence_list[i*batch_size:(i+1)*batch_size] = t.view(batch_size,1)
            true_angle_list[i*batch_size:(i+1)*batch_size] = true_angle
            true_range_list[i*batch_size:(i+1)*batch_size] = true_range
            #Save estimations
            est_presence_list[i*batch_size:(i+1)*batch_size] = max_admap
            est_angle_list[i*batch_size:(i+1)*batch_size] = est_angle
            est_range_list[i*batch_size:(i+1)*batch_size] = est_range

        #Get thresholds that give relatively close to the target Pfa
        init_thr = torch.clone(thresholds)      #To avoid that the next function overwrites the thresholds.
        final_thr = obtainThresholdsFixedPfa(est_presence_list, true_presence_list, target_pfa, delta_pfa, init_thr, device)

        #Lists to save final results
        pd, pfa, rmse_angle, rmse_range, rmse_pos = [], [], [], [], []

        #Compute detection and false alarm probabilities, and RMSEs
        for t in range(len(final_thr)):
            pd_temp, pfa_temp, rmse_angle_temp, rmse_range_temp, rmse_pos_temp = getSensingMetrics(est_presence_list, true_presence_list,
                                                                                                   final_thr[t], est_angle_list,
                                                                                                   true_angle_list, est_range_list,
                                                                                                   true_range_list, device)
            pd.append(pd_temp)
            pfa.append(pfa_temp)
            rmse_angle.append(rmse_angle_temp)
            rmse_range.append(rmse_range_temp)
            rmse_pos.append(rmse_pos_temp)

        #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa (sorting avoids problems with values very similar but not equal)
        pd_inter = np.interp(1e-2,np.sort(pfa),np.sort(pd))
        pfa_inter = np.array(pfa).mean()
        rmse_angle_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_angle))
        rmse_range_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_range))
        rmse_pos_inter = np.interp(1e-2,np.sort(pfa),np.sort(rmse_pos))

    return pd_inter, pfa_inter, rmse_angle_inter, rmse_range_inter, rmse_pos_inter

def testNetworkISAC(sigma_sens, sigma_vector_comm, theta_min_sens_test, theta_max_sens_test, theta_min_comm_test,
                    theta_max_comm_test, range_min_sens_test,
                    range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                    K, S, N0, Delta_f, lamb, true_d, network_d, refConst,
                    rho, phi, target_pfa, delta_pfa, thresholds, batch_size, nTestSamples, device='cpu'):
    '''
    Function to test the sensing performance of a network for a fixed false alarm probability.
    Inputs:
        - sigma_sens: standard deviation of the complex channel gain. non-
        negative Float.
        - sigma_vector_comm: vector of variances for the different taps of the
        communication channel. Float tensor of length S
        - theta_min_sens_test: minimum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - theta_max_sens_test: maximum angle of the coarse angular sector of the
        location of the target. Float, radians.
        - theta_min_comm_test: minimum angle of the coarse angular sector of the
        location of the UE. Float, radians.
        - theta_max_comm_test: maximum angle of the coarse angular sector of the
        location of the UE. Float, radians.
        - range_min_sens_test: minimum range of the coarse range sector of the
        location of the target. non-negative Float.
        - range_max_sens_test: maximum range of the coarse range sector of the
        location of the target. non-negative Float.
        - Ngrid_angle: number of points to perform angular grid search.
        - range_grid: grid of ranges to perform grid search for range estimation.
        Float tensor. The length of range_grid is denoted Ngrid_range.
        - pixels_angle: number of elements in the angle dimension to look around
        the max. of the angle-delay map. Integer.
        - pixels_range: number of elements in the range dimension to look around
        the max. of the angle-delay map. Integer.
        - K: number of antenna elements. Integer.
        - S: number of subcarriers. Integer.
        - N0: variance of the noise added at the receiver. non-negative Float.
        - Delta_f: spacing in Hz between different subcarriers. non-negative Float.
        - lamb: wavelength. non-negative Float.
        - true_d: true inter-antenna spacing to be used in the channel model.
        Float tensor of length K.
        - network_d: assumed inter-antenna spacing by the network. Float tensor
        of length K.
        - refConst: constellation of the transmitted communication symbols.
        Complex Tensor whose length is msg_card.
        - rho: weights to combine the sensing and communication precoders. Float
        list.
        - phi: list of phases to combine the sensing and communication precoders.
        Float list.
        - target_pfa: target false alarm probability (Pfa) to evaluate. Float.
        - delta_pfa: maximum allowed variation of the empirical Pfa with target_pfa.
        Float.
        - thresholds: thresholds to apply to distinguish if there is a target.
        Float tensor of length more than 1 (usually 3).
        - batch_size: size of the data batch. Integer.
        - nTestSamples: number of testing samples. Integer.
        - device: 'cuda' or 'cpu'. Default: 'cpu'.
    Outputs:
        - pd_inter: probability of detection. Float.
        - pfa_inter: probability of false alarm (to check with input). Float.
        - rmse_angle_inter: angle RMSE in rads. Float.
        - rmse_range_inter: range RMSE in meters. Float.
        - rmse_pos_inter: position RMSE in meters. Float.
    '''

    with torch.no_grad():
        #Lists of metrics to save at the end
        pd_total, pfa_total, ser_total = [], [], []
        rmse_angle_total, rmse_range_total, rmse_pos_total = [], [], []

        #Get information from the input data
        msg_card = len(refConst)
        numTestIt = nTestSamples // batch_size
        #Create mean and span values from testing angular sector
        theta_mean_min_sens = theta_mean_max_sens = (theta_min_sens_test+theta_max_sens_test)/2.0
        span_min_theta_sens = span_max_theta_sens = theta_max_sens_test-theta_min_sens_test
        #Create matrix to compute the angle-delay map
        P_matrix, _ = rhoMatrix(range_grid, range_grid, S, Delta_f, device=device)
        theta_mean_min_sens = theta_mean_max_sens = (theta_min_sens_test+theta_max_sens_test)/2.0
        span_min_theta_sens = span_max_theta_sens = theta_max_sens_test-theta_min_sens_test
        theta_mean_min_comm = theta_mean_max_comm = (theta_min_comm_test+theta_max_comm_test)/2.0
        span_min_theta_comm = span_max_theta_comm = theta_max_comm_test-theta_min_comm_test

        #Create radar and communication precoders (learned spacing)
        precoder_sens, _, _, A_matrix, b_matrix_angle = createPrecoder(theta_mean_min_sens, theta_mean_max_sens, \
                                                                    span_min_theta_sens, span_max_theta_sens, K, \
                                                                    batch_size, d=network_d, lamb=lamb, \
                                                                    Ngrid = Ngrid_angle, device=device)
        precoder_sens = precoder_sens[:,0].view(1,-1)
        precoder_comm, _, _, _, _ = createPrecoder(theta_mean_min_comm, theta_mean_max_comm, \
                                                span_min_theta_comm, span_max_theta_comm, K, \
                                                batch_size, d=network_d, lamb=lamb, \
                                                Ngrid = Ngrid_angle, device=device)
        precoder_comm = precoder_comm[:,0].view(1,-1)

        # Create binray matrix (b_size x 1 x N_grid_range) whose columns are 1 for ranges within the considered limits
        b_matrix_range = ((range_grid >= range_min_sens_test.view(1,1,1)) \
                            & (range_grid <= range_max_sens_test.view(1,1,1))).type(torch.cfloat)


        for k in range(len(rho)):
            for l in range(len(phi)):
                #Create lists to save results in each iteration
                true_presence_list = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                true_angle_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                true_range_list    = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                est_presence_list  = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                est_angle_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                est_range_list     = torch.empty(nTestSamples, 1, dtype=torch.float32, device=device)
                true_msg_list      = torch.empty(nTestSamples*S, 1, dtype=torch.float32, device=device)
                est_msg_list       = torch.empty(nTestSamples*S, 1, dtype=torch.float32, device=device)

                #Create precoder for testing
                precoder = createCommonPrecoder(precoder_sens, precoder_comm, rho[k], phi[l], batch_size)
                precoder_rsh = precoder.reshape((batch_size, K, 1))

                for i in range(numTestIt):
                    #Generate random logic vector that tells if there is a target
                    target = torch.randint(0,2,(batch_size,1,1), dtype=torch.float32, device=device)
                    #Generate random messages to transmit
                    msg = createMessages(msg_card, num=batch_size*S, device=device)
                    #Reshape msgs to the needs of the comm CH function
                    symbols = refConst[msg].reshape(batch_size, S, 1)

                    #Sensing channel
                    Y_sens, true_angle, true_range = sensingChannel(sigma_sens, theta_min_sens_test, theta_max_sens_test,
                                                                precoder_rsh, symbols, range_min_sens_test,
                                                                range_max_sens_test, N0, target, Delta_f, lamb,
                                                                true_d, device)

                    #Communication channel
                    y_comm, true_theta_comm, true_beta_comm = commChannel(sigma_vector_comm, theta_min_comm_test, \
                                                                        theta_max_comm_test, K, true_d, lamb, \
                                                                        precoder_rsh, symbols, N0, device)
                    # Sensing Receiver
                    max_admap, est_angle, est_range = sensingReceiver(Y_sens, symbols, A_matrix, b_matrix_angle, \
                                                        b_matrix_range, range_grid, pixels_angle, pixels_range, \
                                                        S, Delta_f, d=network_d, lamb=lamb, device=device)

                    #Comm. receiver
                    kappa = createKappa(true_beta_comm, true_theta_comm, K, true_d, lamb, precoder_rsh, device)
                    est_messages = MLdecoder(y_comm, kappa, refConst)

                    #Save true values
                    true_presence_list[i*batch_size:(i+1)*batch_size] = target.view(batch_size,1)
                    true_angle_list[i*batch_size:(i+1)*batch_size] = true_angle
                    true_range_list[i*batch_size:(i+1)*batch_size] = true_range
                    true_msg_list[i*S*batch_size:(i+1)*S*batch_size] = msg.view(-1,1)
                    #Save estimations
                    est_presence_list[i*batch_size:(i+1)*batch_size] = max_admap[:,0].view(-1,1)    #Here only the 1st column is needed
                    est_angle_list[i*batch_size:(i+1)*batch_size] = est_angle
                    est_range_list[i*batch_size:(i+1)*batch_size] = est_range
                    est_msg_list[i*S*batch_size:(i+1)*S*batch_size] = est_messages.view(-1,1)

                # ================= CALCULATE SER =======================
                num_errors = (true_msg_list != est_msg_list).sum()
                ser = num_errors/(nTestSamples * S)
                #Add SER to vector
                ser_total.append(ser.item())

                # ================= CALCULATE sensing metrics =======================
                #Get thresholds that give relatively close to the target Pfa
                init_thr = torch.clone(thresholds)      #To avoid that the next function overwrites the thresholds.
                final_thr = obtainThresholdsFixedPfa(est_presence_list, true_presence_list, target_pfa, delta_pfa, init_thr, device)

                #Lists to save final results
                pd, pfa, rmse_angle, rmse_range, rmse_pos = [], [], [], [], []

                #Compute detection and false alarm probabilities, and RMSEs
                for t in range(len(final_thr)):
                    pd_temp, pfa_temp, rmse_angle_temp, rmse_range_temp, rmse_pos_temp = getSensingMetrics(est_presence_list, true_presence_list,
                                                                                                        final_thr[t], est_angle_list,
                                                                                                        true_angle_list, est_range_list,
                                                                                                        true_range_list, device)
                    pd.append(pd_temp)
                    pfa.append(pfa_temp)
                    rmse_angle.append(rmse_angle_temp)
                    rmse_range.append(rmse_range_temp)
                    rmse_pos.append(rmse_pos_temp)

                #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa (sorting avoids problems with values very similar but not equal)
                pd_total.append(np.interp(1e-2,np.sort(pfa),np.sort(pd)))
                pfa_total.append(np.array(pfa).mean())
                rmse_angle_total.append(np.interp(1e-2,np.sort(pfa),np.sort(rmse_angle)))
                rmse_range_total.append(np.interp(1e-2,np.sort(pfa),np.sort(rmse_range)))
                rmse_pos_total.append(np.interp(1e-2,np.sort(pfa),np.sort(rmse_pos)))

                print(f'#=== COMPLETED ITERATION {k*len(phi)+l+1}/{len(phi)*len(rho)} ===#', flush=True)

    return pd_total, pfa_total, rmse_angle_total, rmse_range_total, rmse_pos_total, ser_total

"""# Baseline functions"""

def sensingReceiverBaseline(Y_sens, symbols, theta_min, theta_max, r_min, r_max, \
                            f_spacing, d, lamb, numSamplesAngle = 360, \
                            numSamplesRange = 100, device='cpu'):
    '''Function that implements the baseline for the received radar signal,
    to estimate target presence, and if there is a target, its angle and range.
    We assume a MIMO OFDM radar monostatic transceiver with K antennas and S
    subcarriers.
    Inputs:
        - Y_sens: received radar signal. Shape: (batch_size, K, S).
        - symbols: symbols that were transmitted. Shape: (batch_size, S, 1)
        - theta_min: minimum value of the a priori target angle information.
        Real number
        - theta_max: maximum value of the a priori target angle information.
        Real number
        - r_min: minimum of the a priori target range information. Real number
        - r_max: maximum of the a priori target range information. Real number
        - f_spacing: spacing between OFDM subcarriers. Real number
        - d: inter-antenna spacing in the ULA Tx. Real tensor or shape (K,1)
        - lamb: wavelength. Real number
        - numSamplesAngle: number of samples to consider in the grid search for
        angle. Integer. Default: 360
        - numSamplesRange: number of samples to consider in the grid search for
        range. Integer. Default: 100
    Outputs:
        - max_admap: maximum of the angle-delay map of the received signal.
        Shape: (batch_size, 1).
        - est_angle: estimated angle for each possible target in each batch
        sample. Shape: (batch_size, 1)
        - est_range: estimated range for each possible target in each batch
        sample. Shape: (batch_size, 1)
    '''
    batch_size, K, S = Y_sens.shape

    #Remove effect of transmitted comm. symbols
    Y_sens /= symbols.transpose(-1,-2)

    #Grids of angle and range
    theta_vector = torch.linspace(theta_min, theta_max, numSamplesAngle, device = device).reshape(-1,1)
    range_vector = torch.linspace(r_min, r_max, numSamplesRange, device = device).reshape(-1,1)
    tau_vector = 2 * range_vector / 3e8
    # Matrix whose columns are steering vectors for 1 theta
    steeringGrid, _ = steeringMatrix(theta_vector, theta_vector, K, d, lamb, device=device)
    # Matrix whose columns are phase shifts for 1 tau
    phaseGrid, _ = rhoMatrix(range_vector, range_vector, S, f_spacing, device=device)

    #Perform angle-range estimation for all samples to paralellize
    angle_delay_map = torch.abs( torch.conj(steeringGrid.transpose(1,0)) @ Y_sens @ torch.conj(phaseGrid) )
    argmaximum = angle_delay_map.view(batch_size, -1).argmax(dim=-1)           #This avoids loop
    max_theta = torch.div(argmaximum, numSamplesRange, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
    max_range = argmaximum % numSamplesRange
    #We can directly use previous tensors since they are integers
    est_angle = theta_vector[max_theta, 0].view(batch_size, 1)
    est_range = range_vector[max_range, 0].view(batch_size, 1)

    # Take maximum of angle-delay map to later perform detection
    max_admap, _ = torch.max(angle_delay_map.view(batch_size, -1), dim=1, keepdim=True)

    return max_admap, est_angle, est_range

"""# NN functions"""

def sensingReceiver(Y_sens, symbols, A_matrix, b_matrix_angle, \
                       b_matrix_range, range_grid, pixels_angle, pixels_range, \
                       S, f_spacing, d, lamb, device='cpu'):
    '''Function that perform position estimation of a potential target in the
    environment. We assume a MIMO OFDM radar monostatic transceiver
    with K antennas and S subcarriers.
    Inputs:
        - Y_sens: received radar signal. Shape: (batch_size, K, S).
        - symbols: comm. symbols that were transmitted.
        Shape: (batch_size, S, 1).
        - A_matrix: matrix whose columns are steering vectors for a given angle.
        Shape: (K, numSamplesAngle)
        - b_matrix_angle: binary matrix whose columns are 1 for those angles
        inside the prior information and 0 otherwise.
        Shape: (numSamplesAngle, batch_size)
        - P_matrix: matrix whose columns contain phase shifts that convey the
        possible target ranges. Shape: (S, numSamplesRange)
        - b_matrix_range: binary matrix whose columns are 1 for those ranges
        inside the prior information and 0 otherwise.
        - range_grid: oversampled grid of ranges in which to look for the
        targets. Tensor of numSamplesRange elements
        - pixels_angle: number of rows to look around the maximum in the
        metric matrix. Integer tensor
        - pixels_range: number of columns to look around the maximum in the
        metric matrix. Integer tensor
        Shape: (batch_size,1,numSamplesRange)
        - f_spacing: spacing between OFDM subcarriers. Real tensor
        - d: inter-antenna spacing in the ULA Tx. Real tensor or Shape: (K,1)
        - lamb: wavelength. Real tensor
    Outputs:
        - metric_result: estimated metric for each possible target in each batch
        sample. Shape: (batch_size, 1). The metric is the value that
        distinguishes if the measurement is really a target. Last sample does
        not contain target.
        - est_angle_thr: estimated angle for each possible target in each batch
        sample. Shape: (batch_size, 1)
        - est_range_thr: estimated range for each possible target in each batch
        sample. Shape: (batch_size, 1)
    '''
    batch_size, K, S = Y_sens.shape
    numSamplesAngle = A_matrix.shape[1]
    numSamplesRange = len(range_grid)

    #Compute grid of angles to look for
    angle_grid = torch.linspace(-np.pi/2, np.pi/2, numSamplesAngle, device=device)

    #Remove effect of transmitted comm. symbols
    Y_sens /= symbols.transpose(-1,-2)

    #Restrict matrix A to known angle information (each batch sample could have a different angle sector)
    b_matrix_angle_rsh = b_matrix_angle.transpose(1,0).reshape((batch_size, 1, numSamplesAngle))
    A_matrix_con = A_matrix * b_matrix_angle_rsh #Shape: (b_size, K, numSamplesAngle)
    #Compute matrix of delays and restrict matrix P to known range information (each batch sample could have a different range sector)
    P_matrix, _ = rhoMatrix(range_grid, range_grid, S, f_spacing, device)
    P_matrix_con = P_matrix * b_matrix_range #Shape: (b_size, S, Ngrid_range)

    #Vectors to save
    theta_result, range_result = torch.empty(batch_size, 1, device=device), torch.empty(batch_size, 1, device=device)
    metric_result = torch.empty(batch_size, 1, device=device)
    #Matrix to update in each iteration by appending new matrices
    M_vec = torch.tensor([], device=device, dtype=torch.cfloat)
    #Function to maximize (size: batch_size x numSamplesAngle x numSamplesRange)
    metric = torch.abs(A_matrix_con.conj().permute(0,2,1) @ Y_sens @ P_matrix_con.conj())

    '''Estimate angle and range'''
    with torch.no_grad():
        #Compute the index of the highest element of each matrix in the batch, by flattening the matrices first
        _, top_index = torch.topk(metric.view(batch_size,-1), 1, dim=1)
        #Compute the rows and columns of the maximum element
        max_row = torch.div(top_index, numSamplesRange, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
        max_col = top_index % numSamplesRange
        #Create vectors with the corresponding indexes that we want to threshold around the maximum
        max_row_indexes = max_row-pixels_angle + torch.arange(2*pixels_angle+1, dtype=torch.int32, device=device).reshape(1,-1)
        max_col_indexes = max_col-pixels_range + torch.arange(2*pixels_range+1, dtype=torch.int32, device=device).reshape(1,-1)
        #Get the rows and columns of the previous indexes that correspond to values outside the angle-delay map
        temp_mask_rows = torch.where((numSamplesAngle-1 < max_row_indexes) | (max_row_indexes < 0), 1.0, 0.0)
        inf_idx_rows = torch.nonzero(temp_mask_rows, as_tuple=True)
        temp_mask_cols = torch.where((numSamplesRange-1 < max_col_indexes) | (max_col_indexes < 0), 1.0, 0.0)
        inf_idx_cols = torch.nonzero(temp_mask_cols, as_tuple=True)
        #Restrict the possible rows and columns not to exceed the limits of the map
        max_row_restr = torch.maximum(torch.tensor(0, device=device), max_row_indexes)
        max_row_restr = torch.minimum(torch.tensor(numSamplesAngle-1, device=device), max_row_restr)
        max_col_restr = torch.maximum(torch.tensor(0, device=device), max_col_indexes)
        max_col_restr = torch.minimum(torch.tensor(numSamplesRange-1, device=device), max_col_restr)
        #Reshape to take a matrix grid in each batch
        row_sel = max_row_restr.reshape(batch_size, -1,1)
        col_sel = max_col_restr.reshape(batch_size, 1,-1)
    #Select those elements around the maximum in the metric matrix
    metric_thr = metric[torch.arange(batch_size, dtype=torch.long, device=device).reshape(batch_size,1,1), row_sel, col_sel]
    #Substitute the values that correspond to incorrect indexes by -Inf
    #(same as discarding elements, but this way paralellizes operations)
    metric_thr[inf_idx_rows[0], inf_idx_rows[1], :] = torch.tensor(float('-Inf'), device=device)        ########Note: if not using softmax in the next step, be careful with the -Inf
    metric_thr[inf_idx_cols[0], :, inf_idx_cols[1]] = torch.tensor(float('-Inf'), device=device)
    #Apply normalization to the selected elements
    # soft_elements = (metric_thr.view(batch_size, -1) / torch.sum(metric_thr.view(batch_size, -1), dim=1, keepdim=True)).reshape(metric_thr.shape)
    soft_elements = (F.softmax(metric_thr.view(batch_size, -1), dim=1)).reshape(metric_thr.shape)
    #Sum elements corresponding to the same angle or range (row or column)
    soft_row = torch.sum(soft_elements, dim=2).reshape(batch_size, -1)
    soft_col = torch.sum(soft_elements, dim=1).reshape(batch_size, -1)

    #For angle we just care about rows, for range we just care about columns
    angle_grid_con = angle_grid[max_row_restr]
    range_grid_con = range_grid[max_col_restr]
    #Estimate angle and range
    est_angle = torch.sum(soft_row * angle_grid_con, dim=1)
    est_range = torch.sum(soft_col * range_grid_con, dim=1)
    # Compute maximum value of metric to calculate later probability of detection
    metric_max, _ = torch.max(metric.view(batch_size, -1), dim=1)

    #Save estimated angle, range and metric
    metric_result[:,0] = metric_max.flatten()
    theta_result[:,0] = est_angle.flatten()
    range_result[:,0] = est_range.flatten()

    return metric_result, theta_result, range_result