"""
	IntelligentDriverModel <: Policy
The Intelligent Driver Model. A rule based driving model that is governed by parameter
settings. The output is an longitudinal acceleration.
Here, we have extended IDM to the errorable IDM. If a standard deviation parameter is
specified, then the output is a longitudinal acceleration sampled from a normal distribution
around the non-errorable IDM output.
# Fields
- `a::Float64 = NaN` the predicted acceleration i.e. the output of the model
- `σ::Float64 = NaN` allows errorable IDM, optional stdev on top of the model, set to zero or NaN for deterministic behavior
- `k_spd::Float64 = 1.0` proportional constant for speed tracking when in freeflow [s⁻¹]
- `δ::Float64 = 4.0` acceleration exponent
- `T::Float64  = 1.5` desired time headway [s]
- `v_des::Float64 = 29.0` desired speed [m/s]
- `s_min::Float64 = 5.0` minimum acceptable gap [m]
- `a_max::Float64 = 3.0` maximum acceleration ability [m/s²]
- `d_cmf::Float64 = 2.0` comfortable deceleration [m/s²] (positive)
- `d_max::Float64 = 9.0` maximum deceleration [m/s²] (positive)
"""
@with_kw mutable struct IntelligentDriverModel <: Policy
    a::Float64 = 0.0 # predicted acceleration
    σ::Float64 = 0.0 # optional stdev on top of the model, set to zero or NaN for deterministic behavior

    k_spd::Float64 = 1.0 # proportional constant for speed tracking when in freeflow [s⁻¹]

    δ::Float64 = 4.0 # acceleration exponent [-]
    T::Float64  = 1.5 # desired time headway [s]
    v_des::Float64 = 29.0 # desired speed [m/s]
    s_min::Float64 = 5.0 # minimum acceptable gap [m]
    a_max::Float64 = 3.0 # maximum acceleration ability [m/s²]
    d_cmf::Float64 = 2.0 # comfortable deceleration [m/s²] (positive)
    d_max::Float64 = 9.0 # maximum deceleration [m/s²] (positive)
end

function set_desired_speed!(model::IntelligentDriverModel, v_des::Float64)
    model.v_des = v_des
    model
end

# function track_longitudinal!(model::IntelligentDriverModel, v_ego::Float64, v_oth::Float64, headway::Float64)

#     if !isnan(v_oth)
#         #@assert !isnan(headway)
#         if headway < 0.0
#             # @debug("IntelligentDriverModel Warning: IDM received a negative headway $headway"*
#             #       ", a collision may have occured.")
#             model.a = -model.d_max
#         else

#             Δv = v_oth - v_ego
#             s_des = model.s_min + v_ego*model.T - v_ego*Δv / (2*sqrt(model.a_max*model.d_cmf))
#             v_ratio = model.v_des > 0.0 ? (v_ego/model.v_des) : 1.0
#             model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des/headway)^2)
#         end
#     else
#         # no lead vehicle, just drive to match desired speed
#         Δv = model.v_des - v_ego
#         model.a = Δv*model.k_spd # predicted accel to match target speed
#     end

#     #@assert !isnan(model.a)

#     model.a = clamp(model.a, -model.d_max, model.a_max)

#     return model
# end

function track_longitudinal!(model::IntelligentDriverModel, v_ego::Float64, v_oth::Float64, headway::Float64)

    #@assert !isnan(headway)
    if headway < 0.0
        # @debug("IntelligentDriverModel Warning: IDM received a negative headway $headway"*
        #       ", a collision may have occured.")
        model.a = -model.d_max
    else

        Δv = v_oth - v_ego
        s_des = model.s_min + v_ego*model.T - v_ego*Δv / (2*sqrt(model.a_max*model.d_cmf))
        v_ratio = model.v_des > 0.0 ? (v_ego/model.v_des) : 1.0
        model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des/headway)^2)
    end

    model.a = clamp(model.a, -model.d_max, model.a_max)

    return model
end

reset_hidden_state!(model::IntelligentDriverModel) = model


function action(model::IntelligentDriverModel, s_idm::Vector{Float64})
    model = track_longitudinal!(model, s_idm[1], s_idm[2], s_idm[3])
    return model.a
end

function action(model::IntelligentDriverModel, v_ego::Float64)
    Δv = model.v_des - v_ego
    model.a = Δv*model.k_spd # predicted accel to match target speed
    model.a = clamp(model.a, -model.d_max, model.a_max)
    return model.a
end


function Base.rand(rng::AbstractRNG, model::IntelligentDriverModel)
    if isnan(model.σ) || model.σ ≤ 0.0
        return model.a
    else
        rand(rng, Normal(model.a, model.σ))
    end
end

# function Distributions.pdf(model::IntelligentDriverModel, a::LaneFollowingAccel)
#     if isnan(model.σ) || model.σ ≤ 0.0
#         return a == model.a ? Inf : 0.
#     else
#         return pdf(Normal(model.a, model.σ), a.a)
#     end
# end
# function Distributions.logpdf(model::IntelligentDriverModel, a::LaneFollowingAccel)
#     if isnan(model.σ) || model.σ ≤ 0.0
#         return a == model.a ? Inf : 0.
#     else
#         return logpdf(Normal(model.a, model.σ), a.a)
#     end
# end



