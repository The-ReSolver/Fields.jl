# This file contains the recipes required for easy plotting of a channel flow
# at a given time instant.

# TODO: add helper methods for different forms of the field (spectral field, temporal mode snapshot)

@userplot FieldPlot
@recipe function f(p::FieldPlot; arrow_scale=0.025)
    # unpack the arguments
    field, ti = p.args
    y, z, _ = points(get_grid(field))

    # setup the plot
    legend := false
    framestyle := :axes
    xlabel := "\$z\$"
    ylabel := "\$y\$"
    font_size --> 20
    grid := false
    xlims := (0, 2π)
    ylims := (-1, 1)
    colorbar := :right
    colorbar_title --> "\$u^\\prime\$"
    # colorbar_titlefontsize := 15
    colorbar_titlefontvalign := :vcenter

    # contour plot
    @series begin
        seriestype := :contour
        z, y, field[1][:, :, ti]
    end

    # quiver plot
    @series begin
        V = @view(field[2][:, :, ti]); W = @view(field[3][:, :, ti])
        quiver := arrow_scale .* (V[:], W[:])
        seriestype := :quiver
        linecolor --> :black
        repeat(z, inner=length(y)), repeat(y, outer=length(z))
    end
end
