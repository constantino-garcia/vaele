def get_gamma_prior(expected_precision, precision_std):
    beta = expected_precision / precision_std ** 2
    alpha = beta * expected_precision
    return alpha, beta

# get_gamma_prior(float(1e6), float(1e7))

ALPHA = 1e-3
BETA = 1e-3 * 1e-6
AMP = 0.25 ** 2
INITIAL_PREC = 1
DROPOUT = 0
REC_DROPOUT = 0
BIDIRECTIONAL = True
