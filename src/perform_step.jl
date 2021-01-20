# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)

    @unpack t, dt, f = integ
    @unpack d, Proj, SolProj, Precond, du = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache


    @unpack f, p, dt, alg = integ
    @unpack ddu, Proj, Precond, H, u_pred = integ.cache

    # μ = x.μ
    # Σ = x.Σ
    # H = [1 0] * Proj(0)
    # S = H * Σ * H' 
    # K = Σ * H' * inv(S)

    # μ_new = μ + K * (- π / 2 .- H * μ)
    # Σ_new = X_A_Xt(Σ, I - K * H)
    
    integ.cache.x.μ[1] = - π / 2
    integ.cache.x.Σ.squareroot[1, 1] = 0.0
    @info  "" integ.cache.x.μ
    # f(du, Proj(0) * μ_new, p, t)
    # f.jac(ddu, Proj(0) * μ_new, p, t)

    # z = Proj(1) * μ_new - du
    # H = Proj(1) - ddu * Proj(0)

    # S = H * Σ_new * H' 
    # K = Σ_new * H' * inv(S)

    # μ_new2 = μ_new - K * z
    # Σ_new2 = X_A_Xt(Σ_new, I - K * H)

    # @info "" μ_new μ_new2 Σ_new Σ_new2
    # copy!(x, Gaussian(μ_new, Σ_new))

    # @info " " μ_new Σ_new







    # measure!(integ, integ.cache.x, integ.t)

    # x_filt = update!(integ, integ.cache.x)
    # copy!(integ.cache.x, x_filt)

    @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj * cache.x)

end

"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Preconditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache



    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = inv(P)
    x = P * x


    ######################################################################
    # No more dynamic diffusion, because I want to keep it simple for now.
    ######################################################################

    # if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

    #     # Predict
    #     predict_mean!(x_pred, x, A, Q)
    #     mul!(u_pred, SolProj, PI * x_pred.μ)

    #     # Measure
    #     measure!(integ, x_pred, tnew)

    #     # Estimate diffusion
    #     integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
    #     # Adjust prediction and measurement
    #     predict_cov!(x_pred, x, A, apply_diffusion(Q, integ.cache.diffusion))
    #     copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    # else  # Vanilla filtering order: Predict, measure, calibrate

    #     predict!(x_pred, x, A, Q)
    #     mul!(u_pred, SolProj, PI * x_pred.μ)
    #     measure!(integ, x_pred, tnew)
    #     integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
    # end
    ######################################################################

    predict!(x_pred, x, A, Q)


    ##########################################################################################

    # From here on...
    T = pi / 2 # end time step

    # hotfix for weird behaviour in the end region
    if tnew < T 
        x_undone =  PI * x_pred


        P_bvp = Precond(T - tnew)
        PI_bvp = inv(P_bvp)
        
        
        E0 = [1 0] * Proj(0) * PI_bvp


        x_new = P_bvp * x_undone

        meas_mean = E0 * A * x_new.μ - [pi / 2]

        meas_cov = X_A_Xt(x_new.Σ, E0 * A) + X_A_Xt(Q, E0)


        meas_crosscov = x_new.Σ * A' * E0'

        kgain = meas_crosscov * inv(meas_cov)

        μ = x_new.μ - kgain * meas_mean


        h1 = X_A_Xt(x_new.Σ, (I - kgain * E0 * A))
        h2 = X_A_Xt(Q, E0)
        h3 = X_A_Xt(h2, kgain)
        # Σ = h1 + h3

        _L = [h1.squareroot h3.squareroot]
        _, R = qr(_L')
        Σ = SRMatrix(LowerTriangular(collect(R')))

        #     Σ = x_new.Σ + kgain * meas_cov * kgain'
        

        μ_undone = PI_bvp * μ
        # Σ_undone = PI_bvp * Σ
        Σ_undone2 = PI_bvp * Σ.squareroot

        # Back to the old coordinates
        x_pred = Gaussian(μ_undone, SRMatrix(Σ_undone2))


        P = Precond(dt)
        PI = inv(P)

        x_pred = P * copy(x_pred)


    elseif tnew == T 
        
        P = Precond(dt)
        PI = inv(P)

        E0 = [1 0] * Proj(0) * PI


        x_new = x_pred

        meas_mean = E0 * x_new.μ - [pi / 2]

        meas_cov = X_A_Xt(x_new.Σ, E0)


        meas_crosscov = x_new.Σ * E0'

        kgain = meas_crosscov * inv(meas_cov)

        μ = x_new.μ - kgain * meas_mean


        h1 = X_A_Xt(x_new.Σ, (I - kgain * E0))
        # Σ = h1 + h3

        Σ = SRMatrix(h1.squareroot)

        #     Σ = x_new.Σ + kgain * meas_cov * kgain'
        

        μ_undone =  μ
        # Σ_undone = PI_bvp * Σ
        Σ_undone2 = Σ.squareroot

        # Back to the old coordinates
        x_pred = Gaussian(μ_undone, SRMatrix(Σ_undone2))


    end
    ##########################################################################################
    P = Precond(dt)
    PI = inv(P)


    mul!(u_pred, SolProj, PI * x_pred.μ)
    measure!(integ, x_pred, tnew)
    integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)


    # Likelihood
    cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)
    # x_filt = copy(x_pred)


    mul!(u_filt, SolProj, PI * x_filt.μ)
    integ.u .= u_filt

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)
    
    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar

    end

    
    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end


function h!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack u_pred, du, Proj, Precond, measurement = integ.cache
    PI = inv(Precond(dt))
    z = measurement.μ
    E0, E1 = Proj(0), Proj(1)

    u_pred .= E0 * PI * x_pred.μ
    IIP = isinplace(integ.f)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    z .= E1 * PI * x_pred.μ .- du

    return z
end

function H!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack ddu, Proj, Precond, H, u_pred = integ.cache
    E0, E1 = Proj(0), Proj(1)
    PI = inv(Precond(dt))

    if alg isa EK1 || alg isa IEKS
        if alg isa IEKS && !isnothing(alg.linearize_at)
            linearize_at = alg.linearize_at(t).μ
        else
            linearize_at = u_pred
        end

        if isinplace(integ.f)
            f.jac(ddu, linearize_at, p, t)
        else
            ddu .= f.jac(linearize_at, p, t)
            # WIP: Handle Jacobians as OrdinaryDiffEq.jl does
            # J = OrdinaryDiffEq.jacobian((u)-> f(u, p, t), u_pred, integ)
            # @assert J ≈ ddu
        end
        integ.destats.njacs += 1
        mul!(H, (E1 .- ddu * E0), PI)
    else
        mul!(H, E1, PI)
    end

    return H
end


function measure!(integ, x_pred, t)
    @unpack R = integ.cache
    @unpack u_pred, measurement, H = integ.cache

    z, S = measurement.μ, measurement.Σ
    z .= h!(integ, x_pred, t)
    H .= H!(integ, x_pred, t)
    # R .= Diagonal(eps.(z))
    @assert iszero(R)
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return nothing
end


function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))

    return error_estimate
end
