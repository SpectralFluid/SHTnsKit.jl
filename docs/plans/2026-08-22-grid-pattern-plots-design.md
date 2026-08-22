# Grid Pattern Plots for the Web Documentation

## Goal

Help readers understand SHTnsKit's supported latitude grids at a glance by
showing their sampling points on comparable spheres. The visualization should
be attractive, readable at documentation widths, and derived from the package's
actual grid coordinates.

## Visual design

Add a responsive 2-by-2 comparison figure to the Grid Types documentation page.
Every panel uses the same sphere orientation, resolution, scale, and styling so
that the sampling pattern is the only meaningful visual difference.

The panels cover:

1. Gauss–Legendre, with nonuniform latitude rings and no pole samples.
2. Regular midpoint, with uniformly spaced rings offset from both poles.
3. Regular with poles, with uniform spacing and explicit pole samples.
4. Driscoll–Healy, whose pole-inclusive locations match the regular-poles grid.
   Marker size will subtly encode its quadrature weights so the numerical
   distinction remains visible without adding a separate chart.

Each panel includes a concise title and identifying caption. A lightly shaded
sphere, restrained reference lines, and high-contrast sample points preserve
the three-dimensional shape without competing with the grid.

## Generation and integration

A Julia script will construct real `SHTConfig` objects, convert each `(theta,
phi)` pair to Cartesian coordinates, and produce the comparison as a committed
SVG asset. Committing the asset keeps the deployed page independent of plotting
availability during a normal Documenter build. The Grid Types page will embed
the SVG, explain the visible differences, and show minimal constructor examples.
The page will also be added to the Documenter navigation.

## Verification

Regenerate the SVG from a clean invocation, build the documentation, inspect
warnings, and visually check the resulting page at wide and narrow viewport
sizes. Tests should also assert that the generator emits all four titled panels
and uses coordinates obtained from the public constructors.
