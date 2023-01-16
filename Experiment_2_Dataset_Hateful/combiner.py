from dependencies import *

class BaseCalibrator:
    """ Abstract calibrator class
    """
    def __init__(self):
        self.n_classes = None

class TSCalibratorMAP(BaseCalibrator):
    """ MAP Temperature Scaling
    """

    def __init__(self, temperature=1., prior_mu=0.5, prior_sigma=0.5):
        super().__init__()
        self.temperature = temperature
        self.loss_trace = None

        self.prior_mu = torch.tensor(prior_mu)
        self.prior_sigma = torch.tensor(prior_sigma)

    def fit(self, model_logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        _model_logits = torch.from_numpy(model_logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        prior = LogNormal(self.prior_mu, self.prior_sigma)
        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            #print(type(_model_logits),type(_temperature),type(_y))
            loss = nll(_model_logits.type(torch.LongTensor) / _temperature.type(torch.LongTensor), _y.type(torch.LongTensor))
            loss += -1 * prior.log_prob(_temperature)  # This step adds the prior
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  # Temper
        calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)  # Normalize
        return calibrated_probs


class MAPOracleCombiner():
    """ P+L combination method, fit using MAP estimates
    This is our preferred combination method.
    """
    def __init__(self, diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5, calibration_method='temperature scaling'):
        self.confusion_matrix = None  # conf[i, j] is assumed to be P(h = i | Y = j)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50

        self.use_cv = False
        self.calibration_method = calibration_method

        self.calibrator = None
        self.prior_params = {'mu_beta': mu_beta,
                             'sigma_beta': sigma_beta
        }
        #self.n_cls = None
        self.diag_acc = diag_acc
        self.strength = strength

    def fit(self, model_probs, y_h, y_true, num_humans):
        self.n_cls = model_probs.shape[1]
        self.confusion_matrix = []

        # Get MAP estimate of confusion matrix
        for human in range(num_humans):
            alpha, beta = MAPOracleCombiner.get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)

            # alpha values are the diagonal elements of the confusion matrix
            # beta values are the off-diagonal elements of the confusion matrix
            prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta

            # confusion matrix is from the sklearn library
            posterior_matr = 1. * confusion_matrix(y_true, y_h[:,human], labels=np.arange(self.n_cls))
            posterior_matr += prior_matr
            posterior_matr = posterior_matr.T 
            posterior_matr = (posterior_matr - np.ones(self.n_cls)) / (np.sum(posterior_matr, axis=0, keepdims=True) - self.n_cls)
            self.confusion_matrix.append(posterior_matr)

        self.calibrator = TSCalibratorMAP()
        logits = np.log(np.clip(model_probs, 1e-50, 1))
        self.calibrator.fit(logits, y_true)

    def fit_calibrator(self, model_probs, y_true):
        clipped_model_probs = np.clip(model_probs, self.eps, 1)
        model_logits = np.log(clipped_model_probs)
        self.calibrator.fit(model_logits, y_true)

    def fit_calibrator_cv(self, model_probs, y_true):
        # Fits calibration maps that require hyper-parameters, using cross-validation
        if self.calibration_method == 'dirichlet':
            reg_lambda_vals = [10., 1., 0., 5e-1, 1e-1, 1e-2, 1e-3]
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gscv = GridSearchCV(self.calibrator, param_grid={'reg_lambda': reg_lambda_vals,
                                                             'reg_mu': [None]},
                                cv=skf, scoring='neg_log_loss', refit=True)
            gscv.fit(model_probs, y_true)
            self.calibrator = gscv.best_estimator_
        else:
            raise NotImplementedError

    def combine_proba(self, model_probs, y_h, humans):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.shape[0], 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        calibrated_model_probs = self.calibrator.calibrate(model_probs)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = calibrated_model_probs[i]
            for human in humans[i]:
                x = self.confusion_matrix[human]
                y_comb[i] *= x[y_h[i][human]]
            if np.allclose(y_comb[i], 0):  # Handle zero rows
                y_comb[i] = np.ones(self.n_cls) * (1./self.n_cls)

        # Don't forget to normalize :)
        assert np.all(np.isfinite(np.sum(y_comb, axis=1)))
        assert np.all(np.sum(y_comb, axis=1) > 0)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h, humans):
        """ Combines model probs and y_h to return hard labels
        """
        # print(self.confusion_matrix.__len__())
        y_comb_soft = self.combine_proba(model_probs, y_h, humans)
        return np.argmax(y_comb_soft, axis=1)

    def get_dirichlet_params(acc, strength, n_cls):
        # acc: desired off-diagonal accuracy
        # strength: strength of prior

        # Returns alpha,beta where the prior is Dir((beta, beta, . . . , alpha, . . . beta))
        # where the alpha appears for the correct class

        '''
        i think alpha here corresponds to the gamma on page 5's piecewise function
        '''

        beta = 0.1
        alpha = beta * (n_cls - 1) * acc / (1. - acc)

        alpha *= strength
        beta *= strength

        alpha += 1
        beta += 1

        return alpha, beta


# class UnsupervisedEMCombinerMAP(EMCombiner):
#     """ Fully unsupervised EM Combination (fit using MAP estimation)
#     NB: This is referred to in our paper as "P+L-EM"
#     """

#     def __init__(self, calibration_method='MAP temperature scaling', diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5):
#         super().__init__(calibration_method)

#         self.diag_acc = diag_acc
#         self.strength = strength
#         self.prior_alpha = None
#         self.prior_beta = None
#         self.mu_beta = mu_beta
#         self.sigma_beta = sigma_beta

#     def fit(self, model_probs, y_h, num_steps=750):
#         # Initialize
#         self.n_train_u, self.n_cls = model_probs.shape
#         self.prior_alpha, self.prior_beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
#         conf_h = self.initialize_confusion_matrix(self.n_cls)
#         model_probs_clipped = np.clip(model_probs, self.eps, None)
#         model_logits = np.log(model_probs_clipped)
#         calibrated_model_probs = np.copy(model_probs_clipped)

#         # Optimization parameters
#         progbar = tqdm(total=num_steps, leave=False, desc='EM Steps (Unsupervised)')
#         eps = 1e-15  # Clipping parameter to avoid log(0)
#         loss_rel_tol = 1e-6  # Minimum relative change in loss - for early stopping
#         step = 0
#         prev_loss = 1e15
#         loss_tr = []
#         min_steps = 50

#         converged = False
#         while not converged:
#             weight_matrix = self.e_step(calibrated_model_probs, y_h, conf_h)
#             calibrator, conf_h = self.m_step(y_h, model_logits, weight_matrix)

#             # Evaluate loss
#             calibrated_model_probs = calibrator.calibrate(model_probs)
#             calibrated_model_probs_clipped = np.clip(calibrated_model_probs, eps, 1)
#             conf_h_clipped = np.clip(conf_h[y_h], eps, 1)
#             loss = np.sum(weight_matrix * (np.log(calibrated_model_probs_clipped) + np.log(conf_h_clipped)))

#             step += 1
#             if step > num_steps:
#                 warnings.warn('(Unsupervised EM) Maximum number of steps reached -- may not have converged')
#             converged = (step > num_steps) or (np.abs(loss - prev_loss) / np.abs(prev_loss) < loss_rel_tol)
#             if step < min_steps:
#                 converged = False

#             prev_loss = loss
#             loss_tr.append(loss)

#             progbar.update(1)
#         progbar.close()

#         self.calibrator = calibrator
#         self.confusion_matrix = conf_h

#     def m_step(self, y_h, model_logits, weight_matrix):
#         # Get new confusion matrix parameters
#         confusion_matrix = np.empty((self.n_cls, self.n_cls))
#         for b in range(self.n_cls):
#             for a in range(self.n_cls):
#                 # Get entry P(h = a | Y = b)
#                 confusion_matrix[a, b] = weight_matrix[y_h == a, b].sum()
#                 if a == b:
#                     confusion_matrix[a, b] += self.prior_alpha
#                 else:
#                     confusion_matrix[a, b] += self.prior_beta

#         confusion_matrix = np.clip(confusion_matrix, self.eps, None)
#         normalizer = np.sum(confusion_matrix, axis=0, keepdims=True)
#         confusion_matrix = (confusion_matrix - np.eye(self.n_cls)) / (normalizer - self.n_cls)

#         # Get new calibration parameters
#         calibrator = self.get_calibrator(mu_beta=self.mu_beta, sigma_beta=self.sigma_beta)
#         calibrator.fit(model_logits, weight_matrix)

#         return calibrator, confusion_matrix