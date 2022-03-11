using AutoViz
using AutomotiveDrivingModels
using Cairo

struct CrosswalkEnv
    roadway::Roadway{Float64}
    crosswalk::Lane{Float64}
end

roadway_length = 30.
crosswalk_length = 12.
crosswalk_width = 4.0
crosswalk_pos = 25

# # Generate a straight 2-lane roadway and a crosswalk lane
roadway = gen_straight_roadway(2, roadway_length)
crosswalk_start = VecE2(crosswalk_pos, -crosswalk_length/2)
crosswalk_end = VecE2(crosswalk_pos, crosswalk_length/2)
crosswalk_lane = gen_straight_curve(crosswalk_start, crosswalk_end, 2)
crosswalk = Lane(LaneTag(2,1), crosswalk_lane, width = crosswalk_width)
cw_segment = RoadSegment(2, [crosswalk])
push!(roadway.segments, cw_segment) # append it to the roadway


# initialize crosswalk environment
env = CrosswalkEnv(roadway, crosswalk)

function AutoViz.add_renderable!(rendermodel::RenderModel, env::CrosswalkEnv)

    # render the road without the crosswalk
    roadway = gen_straight_roadway(2, roadway_length)
    add_renderable!(rendermodel, roadway)

    # render crosswalk
    curve = env.crosswalk.curve
    n = length(curve)
    pts = Array{Float64}(undef, 2, n)
    for (i,pt) in enumerate(curve)
        pts[1,i] = pt.pos.x
        pts[2,i] = pt.pos.y+1.5
    end

    pts2 = copy(pts)
    pts2[2, 1] = pts2[2, 1] .+ 1.0
    #pts2[1, :] = pts2[2, :] .- 1.0
    add_instruction!(
        rendermodel, render_dashed_line,
        (pts[:, 1:end], colorant"white", env.crosswalk.width, 1.0, 1.0, 0.0, Cairo.CAIRO_LINE_CAP_BUTT)
    )
    add_instruction!(
        rendermodel, render_dashed_line,
        (pts2, colorant"0x708090", env.crosswalk.width, 1.0, 1.0, 0.0, Cairo.CAIRO_LINE_CAP_BUTT)
    )
    return rendermodel
end

snapshot = render([env])

const PEDESTRIAN_DEF = VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0)

# Car definition
car_initial_state = VehicleState(VecSE2(5.0, 0., 0.), roadway.segments[1].lanes[1],roadway, 8.0)
car = Entity(car_initial_state, VehicleDef(), :car)

# Pedestrian definition using our new Vehicle definition
ped_initial_state = VehicleState(VecSE2(+25.0,-4.0,π/2), env.crosswalk, roadway, 0.5)
ped = Entity(ped_initial_state, PEDESTRIAN_DEF, :pedestrian)

scene = Scene([car, ped])

# renderables = [
#     roadway,
#     (ArrowCar(scene[i]) for i in 1:3)...,
#     FancyPedestrian(ped=scene[4])
# ]

#renderables = [env, ArrowCar(scene[1], color=colorant"green"), scene[2]]
renderables = [env, ArrowCar(scene[1], color=colorant"black"), EntityRectangle(entity=scene[2], color=colorant"#8e474a"),  VelocityArrow(entity=scene[2], color=colorant"white")]

# visualize the initial state
# snapshot = render([env, scene])
snapshot = render(renderables)
write_to_png(render(renderables), "./data/crosswalk.png")