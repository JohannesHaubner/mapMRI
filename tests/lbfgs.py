from scipy.optimize import minimize, rosen, rosen_der

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

res = minimize(rosen, x0, method='L-BFGS-B', tol=1e-6, options={"iprint": 99})

res.x