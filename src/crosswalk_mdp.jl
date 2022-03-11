
# MDP definition
mutable struct AdversarialCrosswalkMDP{P<:Policy} <: MDP{Vector{Float64}, Vector{Float64}}
    sut_policy::P # 
    #agents::Vector{Agent} # All the agents ordered by (adversaries..., sut, others...)
    #vehid2ind::Dict{Int64, Int64} # Dictionary that maps vehid to index in agent list
    #num_adversaries::Int64 # The number of adversaries
    # roadway::Roadway # The roadway for the simulation
    dt::Float64 # Simulation timestep
    #last_observation::Array{Float64} # Last observation of the vehicle state
    #disturbance_model # Model used for disturbances. supports `logpdf` and `rand` and `actions`
    γ::Float64 # discount
    #ast_reward::Bool # A function that gives action log prob.
    #no_collision_penalty::Float64 # penalty for not getting a collision (for ast reward)
    #scale_reward::Bool #whether or not to scale the AST reward
    #end_of_road::Float64 # specify an early end of the road
    crosswalk_width::Float64
    lane_width::Float64
end

# Contrsuctor
# AdversarialCrosswalkMDP(problem::Union{POMDP,MDP};
#              rng=Random.GLOBAL_RNG,
#              updater=NothingUpdater()) = RandomPolicy(rng, problem, updater)

function observe_pedestrian(mdp::AdversarialCrosswalkMDP, s::Vector{Float64}, x::Vector{Float64})
    s_ped = s[3:end]
    return vcat(s_ped[1:2] .+ x[1:2], s_ped[3:4] .+ x[3:4])
end

function action_sut(mdp::AdversarialCrosswalkMDP, s::Vector{Float64}, x::Vector{Float64})
    o_ped = observe_pedestrian(mdp, s, x)
    
    x_ego = s[1]
    vx_ego = s[2]
    #v_ped = sqrt(observed_s_ped[3]^2 + observed_s_ped[4]^2)
    vx_ped = o_ped[3]
    x_ped = o_ped[1]
    
    # if pedestrian in road headway
    # else headway = 0
    #headway = clamp(x_ped - x_ego, 0, Inf)
    if o_ped[2] > -mdp.lane_width/2 && o_ped[2] < mdp.lane_width/2 && x_ped >= x_ego
        # @show x_ped - x_ego
        # y_ped = o_ped[2]
        # @show y_ped
        headway = x_ped - x_ego
        #clamp(headway, 0, Inf)
        a_ego = action(mdp.sut_policy, [vx_ego, vx_ped, headway])
    else
        a_ego = action(mdp.sut_policy, vx_ego)
    end

    return a_ego

end


function action_ped(mdp::AdversarialCrosswalkMDP, s::Vector{Float64}, x::Vector{Float64})
    #a_ped_new = s[9:10] + x[1:2]
    #return vcat(s[1:8], a_ped_new)
    a_ped_new = x[5:6]
    return a_ped_new

end


function propagate(mdp::AdversarialCrosswalkMDP, s::Vector{Float64})
    dt = mdp.dt
    s_ped = s[4:7]
    #a_ped = s_ped[5:end]
    x_ped = s_ped[1:2]
    v_ped = s_ped[3:4]
    v_ped_new = v_ped .+ a_ped*dt
    x_ped_new = x_ped .+ v_ped*dt .+ a_ped*dt^2

    s_ego = s[1:3]
    v_ego_new = s_ego[2] + s_ego[3]*dt
    x_ego_new = s_ego[1] + s_ego[2]*dt + s_ego[3]*dt^2

    s_new = vcat(x_ego_new, v_ego_new, s_ego[3], x_ped_new, v_ped_new, a_ped)
    return s_new
end

function step!(mdp::AdversarialCrosswalkMDP, s::Vector{Float64}, x::Vector{Float64})
    # Extract ego state
    # ego_state = s[1:3]
    
    # ped_state = s[4:6]

    a_ped = action_ped(mdp, s, x)
    a_ego = action_sut(mdp, s, x)

    #state = propagate(mdp, sut_updated_state)

    dt = mdp.dt
    s_ped = s[3:6]
    x_ped = s_ped[1:2]
    v_ped = s_ped[3:4]
    v_ped_new = v_ped .+ a_ped*dt
    #x_ped_new = x_ped .+ v_ped*dt .+ a_ped*dt^2
    x_ped_new = x_ped .+ v_ped*dt #.+ 0.5*a_ped*dt^2

    s_ego = s[1:2]
    v_ego_new = s_ego[2] + a_ego*dt
    if v_ego_new < 0
        v_ego_new = 0
        a_ego = 0
        s_ego = [s[1], 0.0]
    end
    #x_ego_new = s_ego[1] + s_ego[2]*dt + a_ego*dt^2
    #x_ego_new = s_ego[1] + s_ego[2]*dt + a_ego*dt^2
    x_ego_new = s_ego[1] + s_ego[2]*dt# + 0.5*a_ego*dt^2

    s_new = vcat(x_ego_new, v_ego_new, x_ped_new, v_ped_new)
    return s_new


end



