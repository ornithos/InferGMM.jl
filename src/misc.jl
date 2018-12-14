dropdim1(x) = dropdims(x, dims=1)   # useful for pipes
dropdim2(x) = dropdims(x, dims=2)

eye(d) = Matrix(I, d, d)