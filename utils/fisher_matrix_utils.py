# from utils.generic_utils import *
#
#
# # # FIXME: all representation of psi_0
# # def gamma_block_of_fisher(psi_0, psi_1):
# #     F_00 = 4 * (psi_0 ** 2) * T.tri_gamma(1 + psi_0 ** 2)
# #     F_01 = -4 * psi_0 / psi_1
# #     F_11 = 4 * (1 + psi_0 ** 2) / (psi_1 ** 2)
# #     return T.stack([T.stack([F_00, F_01]),
# #                     T.stack([F_01, F_11])]
# #                    )
#
# def inv_fisher_gamma_block(psi_0, psi_1, epsilon=1e-9):
#     F_00 = 4 * (psi_0 ** 2) * T.tri_gamma(psi_0 ** 2)
#     F_01 = -4 * psi_0 / psi_1
#     F_11 = 4 * (psi_0 ** 2) / (psi_1 ** 2)
#     # Ensure positive definite matrix by adding a small quantity to the diagonal:
#     F_00 = F_00 + epsilon
#     F_11 = F_11 + epsilon
#     det = F_00 * F_11 - F_01 ** 2
#     return 1 / det * T.stack(
#         [T.stack([F_11, -F_01]), T.stack([-F_01, F_00])]
#     )
#
#
# def calculate_fisher_22(alpha_db_beta, sigma):
#     return alpha_db_beta * sigma
#
# def calculate_fisher_22_p2(alpha_db_beta, precision):
#     return alpha_db_beta * precision
#
# def calculate_fisher_23(alpha_db_beta, cov_mean_tensor3, psi_3_dot_psi_4, cholesky_indices):
#     transformed_tensor3 = cov_mean_tensor3 + cov_mean_tensor3.dimshuffle(2, 1, 0)
#     results, _ = theano.scan(
#         lambda row, column, alpha_db_beta, auxiliar_tensor, auxiliar_mat: (
#             -2 * alpha_db_beta * (
#                 T.dot(auxiliar_tensor[row, :, :], auxiliar_mat[:, column])
#             )
#         ),
#         sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]],
#         non_sequences=[alpha_db_beta, transformed_tensor3, psi_3_dot_psi_4]
#     )
#     return results.T
#
#
# def calculate_fisher_23_p2(dimension):
#     return T.zeros((dimension, (dimension * (dimension - 1)) // 2))
#
# def calculate_fisher_24(alpha_db_beta, cov_mean_tensor3, psi_3_matrix, psi_4):
#     transformed_tensor3 = cov_mean_tensor3 + cov_mean_tensor3.dimshuffle(0, 2, 1)
#     results, _ = theano.scan(
#         lambda vec, exp_psi_4_i, aux_tensor, alpha_db_beta: (
#             -alpha_db_beta * T.tensordot(T.tensordot(aux_tensor, vec.flatten(), axes=[[2], [0]]), vec.flatten(),
#                                          axes=[[1], [0]]) * exp_psi_4_i
#
#         ),
#         sequences=[psi_3_matrix.T, T.exp(psi_4)],
#         non_sequences=[transformed_tensor3, alpha_db_beta]
#     )
#     return results.T
#
# def calculate_fisher_24_p2(dimension):
#     return T.zeros((dimension, dimension))
#
# def calculate_fisher_33(fourth_moment_tensor4, psi_3_dot_psi_4, cholesky_indices):
#     def F_3_3_with_the_first_fixed(i, j, psi_3_dot_psi_4, fourth_moment_tensor4, cholesky_indices):
#         results, _ = theano.scan(
#             lambda k, l, i, j, psi_3_dot_psi_4, fourth_moment_tensor4: (
#                 4 * T.dot(T.dot(fourth_moment_tensor4[i, :, k, :], psi_3_dot_psi_4[:, l]),
#                           psi_3_dot_psi_4[:, j])
#             ),
#             sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]],
#             non_sequences=[i, j, psi_3_dot_psi_4, fourth_moment_tensor4]
#         )
#         return results.flatten()
#
#     results, _ = theano.scan(
#         F_3_3_with_the_first_fixed,
#         sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]],
#         non_sequences=[psi_3_dot_psi_4, fourth_moment_tensor4, cholesky_indices]
#     )
#     return results
#
# def calculate_fisher_33_p2(psi_3_inv_matrix, psi_4, cov, cholesky_indices):
#     def F_3_3_with_the_first_fixed(j, i, psi_3_inv_matrix, psi_4, cov, cholesky_indices):
#         results, _ = theano.scan(
#             lambda l, k: (
#                 psi_3_inv_matrix[i, j] * psi_3_inv_matrix[k, l] +
#                 psi_3_inv_matrix[i, l] * psi_3_inv_matrix[k, j] +
#                 2 * T.switch(T.eq(i, k), 1, 0) * cov[j, l] * T.exp(psi_4[i])
#             ),
#             sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]]
#         )
#         return results.flatten()
#
#     results, _ = theano.scan(
#         lambda i, j: F_3_3_with_the_first_fixed(i, j, psi_3_inv_matrix, psi_4, cov, cholesky_indices),
#         sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]]
#     )
#     return results
#
#
#
# def calculate_fisher_34(fourth_moment_tensor4, psi_3_matrix, psi_3_dot_psi_4, cholesky_indices):
#     def F_3_4_with_the_first_fixed(i, j, psi_3_matrix, psi_3_dot_psi_4, fourth_moment_tensor4):
#         auxiliar_matrix = T.tensordot(fourth_moment_tensor4[i, :, :, :], psi_3_dot_psi_4[:, j], axes=[[0], [0]])
#         results, _ = theano.scan(
#             lambda psi_3_k_column, psi_3_dot_psi_4_k_column, auxiliar_matrix: (
#                 2 * T.dot(T.dot(auxiliar_matrix, psi_3_k_column), psi_3_dot_psi_4_k_column)
#             ),
#             sequences=[psi_3_matrix.T, psi_3_dot_psi_4.T],
#             non_sequences=[auxiliar_matrix]
#         )
#         return results.flatten()
#
#     results, _ = theano.scan(
#         F_3_4_with_the_first_fixed,
#         sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]],
#         non_sequences=[psi_3_matrix, psi_3_dot_psi_4, fourth_moment_tensor4]
#     )
#     return results
#
#
# def calculate_fisher_34_p2(psi_3_inv_matrix, cholesky_indices, dimension):
#     z = T.arange(dimension)
#     results, _ = theano.scan(
#         lambda j, i: psi_3_inv_matrix[i, j] * T.switch(T.eq(z, i), 1, 0),
#         sequences=[cholesky_indices[:, 0], cholesky_indices[:, 1]],
#     )
#     return results
#
#
# def calculate_fisher_44(cov, fourth_moment_tensor4, psi_3_matrix, psi_4):
#     psi_3_products, _ = theano.scan(
#         lambda psi_3_column: T.dot(psi_3_column.reshape((-1, 1)), psi_3_column.reshape((1, -1))),
#         sequences=[psi_3_matrix.T]
#     )
#     diagonal_fisher, _ = theano.scan(
#         lambda psi_3_product_i, exp_psi_4_i, cov, fourth_moment_tensor4: (
#             exp_psi_4_i ** 2 * T.tensordot(psi_3_product_i,
#                                            T.tensordot(fourth_moment_tensor4, psi_3_product_i)) +
#             -T.sum(psi_3_product_i * cov) * exp_psi_4_i + 0.25
#         ),
#         sequences=[psi_3_products, T.exp(psi_4)],
#         non_sequences=[cov, fourth_moment_tensor4]
#     )
#     return T.diag(diagonal_fisher)
#
#
# def calculate_fisher_44_p2(cov):
#    return 0.5 * T.identity_like(cov)
#
#
# def calculate_inverse_fisher_44_p2(cov):
#    return 2 * T.identity_like(cov)
#
#
# def calculate_inv_fisher_blocks(alpha, beta, mean, cov, psi_0, psi_1, psi_3, psi_4, dimension):
#     # calculate some useful variables and tensors
#     cholesky_indices = theano.shared(get_cholesky_indices(dimension).astype('int32'))
#     psi_3_matrix = vector_to_tril_theano(psi_3, dimension)
#     psi_3_dot_psi_4 = T.dot(psi_3_matrix, T.diag(T.exp(psi_4)))
#     alpha_db_beta = alpha / beta
#     # Expectation([x1 * x2 - mu1 * mu2] * [x3 * x4 - mu3 * mu4]) = T_{1,2,3,4}
#     mean_matrix = T.dot(mean.reshape((-1, 1)), mean.reshape((1, -1)))
#     covcov_tensor = T.tensordot(cov.dimshuffle(0, 1, 'x'), cov.dimshuffle('x', 0, 1), [[2], [0]])
#     mean_mean_cov_tensor4 = T.tensordot(mean_matrix.dimshuffle(0, 1, 'x'), cov.dimshuffle('x', 0, 1),
#                                         [[2], [0]])
#     fourth_moment_tensor4_1 = (
#         mean_mean_cov_tensor4.dimshuffle(2, 0, 3, 1) + mean_mean_cov_tensor4.dimshuffle(2, 0, 1, 3) +
#         mean_mean_cov_tensor4.dimshuffle(0, 2, 3, 1) + mean_mean_cov_tensor4.dimshuffle(0, 2, 1, 3))
#     fourth_moment_tensor4_2 = (covcov_tensor.dimshuffle(0, 2, 3, 1) + covcov_tensor.dimshuffle(0, 2, 1, 3) +
#                                covcov_tensor.dimshuffle(0, 1, 2, 3)
#                                )
#     fourth_moment_tensor4 = alpha_db_beta * fourth_moment_tensor4_1 + fourth_moment_tensor4_2
#
#     cov_mean_tensor3 = T.tensordot(cov.dimshuffle(0, 1, 'x'), mean.dimshuffle('x', 0), axes=[[2], [0]])
#
#     inv_fisher_block_0 = inv_fisher_gamma_block(psi_0, psi_1)
#     f_22 = calculate_fisher_22(alpha_db_beta, cov)
#     f_23 = calculate_fisher_23(alpha_db_beta, cov_mean_tensor3, psi_3_dot_psi_4, cholesky_indices)
#     f_24 = calculate_fisher_24(alpha_db_beta, cov_mean_tensor3, psi_3_matrix, psi_4)
#     f_33 = calculate_fisher_33(fourth_moment_tensor4, psi_3_dot_psi_4, cholesky_indices)
#     f_34 = calculate_fisher_34(fourth_moment_tensor4, psi_3_matrix, psi_3_dot_psi_4, cholesky_indices)
#     f_44 = calculate_fisher_44(cov, fourth_moment_tensor4, psi_3_matrix, psi_4)
#
#     fisher_block_1 = T.concatenate([
#         T.concatenate([f_22, f_23, f_24], axis=1),
#         T.concatenate([f_23.T, f_33, f_34], axis=1),
#         T.concatenate([f_24.T, f_34.T, f_44], axis=1)
#     ], axis=0)
#
#     inv_fisher_block_1 = T.nlinalg.pinv(fisher_block_1)
#     inv_fisher_block_1 = (inv_fisher_block_1 + inv_fisher_block_1.T) / 2.0
#
#     return inv_fisher_block_0, inv_fisher_block_1
#
#
# def calculate_general_inv_fisher_blocks_p2(alpha, beta, mean, cov, psi_0, psi_1, psi_3, psi_4, dimension):
#     precision = T.nlinalg.pinv(cov)
#     # calculate some useful variables and tensors
#     cholesky_indices = theano.shared(get_cholesky_indices(dimension).astype('int32'))
#     psi_3_matrix = vector_to_tril_theano(psi_3, dimension)
#     psi_3_inv_matrix = T.nlinalg.pinv(psi_3_matrix)
#
#     psi_3_dot_psi_4 = T.dot(psi_3_matrix, T.diag(T.exp(psi_4)))
#     alpha_db_beta = alpha / beta
#     # Expectation([x1 * x2 - mu1 * mu2] * [x3 * x4 - mu3 * mu4]) = T_{1,2,3,4}
#     mean_matrix = T.dot(mean.reshape((-1, 1)), mean.reshape((1, -1)))
#     covcov_tensor = T.tensordot(cov.dimshuffle(0, 1, 'x'), cov.dimshuffle('x', 0, 1), [[2], [0]])
#     mean_mean_cov_tensor4 = T.tensordot(mean_matrix.dimshuffle(0, 1, 'x'), cov.dimshuffle('x', 0, 1),
#                                         [[2], [0]])
#     fourth_moment_tensor4_1 = (
#         mean_mean_cov_tensor4.dimshuffle(2, 0, 3, 1) + mean_mean_cov_tensor4.dimshuffle(2, 0, 1, 3) +
#         mean_mean_cov_tensor4.dimshuffle(0, 2, 3, 1) + mean_mean_cov_tensor4.dimshuffle(0, 2, 1, 3))
#     fourth_moment_tensor4_2 = (covcov_tensor.dimshuffle(0, 2, 3, 1) + covcov_tensor.dimshuffle(0, 2, 1, 3) +
#                                covcov_tensor.dimshuffle(0, 1, 2, 3)
#                                )
#     fourth_moment_tensor4 = alpha_db_beta * fourth_moment_tensor4_1 + fourth_moment_tensor4_2
#
#     cov_mean_tensor3 = T.tensordot(cov.dimshuffle(0, 1, 'x'), mean.dimshuffle('x', 0), axes=[[2], [0]])
#
#     inv_fisher_block_0 = inv_fisher_gamma_block(psi_0, psi_1)
#     f_22 = calculate_fisher_22_p2(alpha_db_beta, precision)
#     f_23 = calculate_fisher_23_p2(dimension)
#     f_24 = calculate_fisher_24_p2(dimension)
#     f_33 = calculate_fisher_33_p2(psi_3_inv_matrix, psi_4, cov, cholesky_indices)
#     # TODO: check if this is always 00000!!
#     f_34 = calculate_fisher_34_p2(psi_3_inv_matrix, cholesky_indices, dimension)
#     f_44 = calculate_fisher_44_p2(cov)
#
#     fisher_block_1 = T.concatenate([
#         T.concatenate([f_22, f_23, f_24], axis=1),
#         T.concatenate([f_23.T, f_33, f_34], axis=1),
#         T.concatenate([f_24.T, f_34.T, f_44], axis=1)
#     ], axis=0)
#
#     inv_fisher_block_1 = T.nlinalg.pinv(fisher_block_1)
#     inv_fisher_block_1 = (inv_fisher_block_1 + inv_fisher_block_1.T) / 2.0
#
#     return inv_fisher_block_0, inv_fisher_block_1
#
#
# def calculate_blocked_inv_fisher_p2(alpha, beta, mean, cov, psi_0, psi_1, psi_3, psi_4, dimension):
#     # calculate some useful variables and tensors
#     precision = T.nlinalg.pinv(cov)
#     cholesky_indices = theano.shared(get_cholesky_indices(dimension).astype('int32'))
#     psi_3_matrix = vector_to_tril_theano(psi_3, dimension)
#     psi_3_inv_matrix = T.nlinalg.pinv(psi_3_matrix)
#     alpha_db_beta = alpha / beta
#
#     inv_fisher_block_0 = inv_fisher_gamma_block(psi_0, psi_1)
#     inv_fisher_block_mean = T.nlinalg.matrix_inverse(
#         calculate_fisher_22_p2(alpha_db_beta, precision)
#     )
#     inv_fisher_block_mean = (inv_fisher_block_mean + inv_fisher_block_mean.T) / 2.0
#     inv_fisher_block_L = T.nlinalg.matrix_inverse(
#         calculate_fisher_33_p2(psi_3_inv_matrix, psi_4, cov, cholesky_indices)
#     )
#     inv_fisher_block_L = (inv_fisher_block_L + inv_fisher_block_L.T) / 2.0
#     inv_fisher_block_logD = calculate_inverse_fisher_44_p2(cov)
#
#     return inv_fisher_block_0, inv_fisher_block_mean, inv_fisher_block_L, inv_fisher_block_logD