nb_sims = 16
min_A = 0.3
max_A = 1

root_folder = "csv"

for (mode in c('train', 'test')) {
  folder = file.path(root_folder, mode)

  Ts = 0.01
  time = seq(0, len = 70, by = Ts)
  omega = 2 * pi * 3

  if (mode == 'train') {
    As = seq(min_A, max_A, length.out = nb_sims)
  } else {
    As = runif(nb_sims, min_A, max_A)
  }
  for (i in 1:nb_sims) {
    phi = runif(1, 0, 2 * pi)
    A = As[i]
    x = A * cos(omega * time + phi)
    write.table(x, file.path(folder, paste0("y_", i, ".csv")), quote = FALSE,
              col.names = FALSE, row.names = FALSE)
  }
}
