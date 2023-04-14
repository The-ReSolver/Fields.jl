# This file contains the recipes required for easy plotting of a channel flow
# at a given time instant.

# TODO: add helper methods for different forms of the field (spectral field, temporal mode snapshot)

@userplot FieldPlot
@recipe function f(p::FieldPlot; arrow_scale=0.025)
    # unpack the arguments
    field, ti = p.args
    y, z, _ = points(get_grid(field)); z = collect(z)

    # append field for periodicity
    extended_field = Vector{typeof(parent((field[1])))}(undef, length(field))
    for i in eachindex(field)
        extended_field[i] = cat(parent(field[i]), field[i][:, 1:1, :]; dims=2)
    end
    push!(z, 2π)

    # define font
    font = 

    # setup the plot
    legend := false
    framestyle := :axes
    xlabel --> "\$z\$"
    ylabel -->"\$y\$"
    guidefont --> ("serif", 20)
    tickfontsize --> 15
    tick_direction --> :out
    grid := false
    xlims := (0, 2π)
    ylims := (-1.03, 1.03)
    colorbar := :right
    colorbar_title --> "\$u^\\prime\$"
    colorbar_titlefont --> ("serif", 20)
    colorbar_tickfontsize --> 15

    # contour plot
    @series begin
        seriestype := :contourf
        color --> :jet
        levels --> 40
        z, y, extended_field[1][:, :, ti]
    end

    # quiver plot
    @series begin
        V = @view(extended_field[2][:, :, ti]); W = @view(extended_field[3][:, :, ti])
        quiver := arrow_scale .* (V[:], W[:])
        seriestype := :quiver
        linecolor --> :black
        arrow --> (:closed, :head, (0.2, 0.2)) # ! ARROWHEAD SIZE SEEMS TO NOT BE SUPPORTED IN GR
        repeat(z, inner=length(y)), repeat(y, outer=length(z))
    end

    # top wall
    @series begin
        seriestype := :path
        linecolor := nothing
        fillcolor --> :darkgrey
        fillalpha --> 1.0
        fillrange := 1.03
        markershape := :none
        z, repeat([1.0], inner=length(z))
    end

    # bottom wall
    @series begin
        seriestype := :path
        linecolor := nothing
        fillcolor --> :darkgrey
        fillalpha --> 1.0
        fillrange := -1.03
        markershape := :none
        z, repeat([-1.0], inner=length(z))
    end
end
