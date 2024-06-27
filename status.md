Features/ToDo
=============

- [x] Basic radiance cascade method brought from python impl (@22299f7)
- [x] Update raymarcher to Amanatides & Woo formalism (@fcdf129)
- [x] Reformulate for flatland participating media (@c808b28)
- [x] Convert to 2.5D (@c808b28)
- [x] Track down issues on top edge (@95994d0)
- [x] Compare with Lightweaver on simple case (FS only) (@7beb0fb)
- [x] Add mipmapping for upper levels (@5f65d5c)
- [x] Add branching/bilinear fix and compare (@5f65d5c)
- [x] CRTAF parsing (@ed49231)
- [x] Load atmosphere (@85ed408)
- [x] LTE populations (@b54f9ff)
- [x] Line absorption profile (@c878288)
- [x] Emission, opacity, source function (@2bf9af7 - not source function, but that's just a ratio!)
- [x] LTE formal solution from atmosphere (@bbddad4 - static only)
- [ ] LTE background from tabulated atmosphere
- [ ] LTE with background J iteration
- [x] Final formal solution (interactive?)
    - Mostly working. EmisOpac system needs to be refactored to handle "random" wavelengths
- [x] Non-LTE iteration (@5a68d12)
    - [x] Add line profile normalisation in final cascade (`wphi`)
- [ ] Boundary conditions
    - [x] Interpolated Prom style
        - Full 3D treatment available (but only considering ray projection in
        x-z), applying a correction factor for solid angle due to muz
    - [ ] Periodic
        - Lagged sampling from previous cascade 0? -- very diffusive
        - Ping-pong full cascade stack and do a directional lookup/traversal
            - Could do this in a separate alg and build a BC from this
        - Hauschildt & Baron style multiple wrap-around?
    - [x] Make generic via templating struct passed down to raymarcher -- each BC
    has a `sample` function so dispatch is fully static (there won't be many
    types so use a switch to template dispatch). Needs some tidying to wrap up exploding cases as BCs are added
- [x] Multiple atoms - basically splat everything SOA a la MULTI and like is done for the one atom here. -- Messy, but done
- [x] Charge conservation
    - [ ] Finite difference charge conservation blows up if we're off the end of the grid -- makes sense; dC will also explode
- [x] Pressure conservation
- [x] Config file/input flags
- [ ] Pydantic objects for input files
    - [x] Dexrt
    - [ ] Ray
- [x] Active atmosphere cells (temperature criterion)
- [x] Active probes (from C0 active cells)
- [ ] Save compiled params into output
    - [ ] Add attributes to netcdf layer
- [ ] Add groups to netcdf layer
- [ ] Restart from snapshot
- [ ] Embed git hash in build (https://jonathanhamberg.com/post/cmake-embedding-git-hash/)
- [ ] PRD
- [ ] Sparse VDB-like grid


Ideas
=====

- [ ] Move LTE populations to be partition function based (allows for calculation of only the levels of interest)
- [x] Use magma to abstract batched LU solved for populations
    - [ ] Needs non-GPU alternative
- [x] Store rays per cascade pre-averaged (groups of 4) - should reduce bandwidth and storage requirements, but requires atomic operation (available in refactor)
- [x] Avoid local memory in dda ray traversal -- seems to be causing stalls
    - Implemented fully register-based approach using the switch/template method from nanovdb - big perf difference.
- [x] Full per-wavelength active set treatment. GPU benefits a lot more from this
- [x] Bring back wavelength batches -- consider a warp (32 threads) of inclination rays (e.g. 4) with e.g. 8 wavelengths. The raymarching will be entirely coherent for these. Emissivity/Opacity gather will be almost perfectly coherent too. In 3D, if we have memory do full warps of wavelengths to get this coherence back.
- [x] Refactor to only have one raymarch/RC impl
- [ ] Handle case of solving one direction of c0 at a time (with all necessary components of upper cascades) - same memory as preaveraging, but useful for e.g. dynamic models in 3D
- [ ] Is it possible to create a basis of emissivity and opacity that can be interpolated as a function of mux/muy/muz to allow emissivity/opacity to be computed for fewer directions, mipmapped, and then linearly combined in-situ? Needs tests


Notes
=====

Formalising coordinate system
------------------------------
- z up, x default perpendicular axis, y as per right-hand rule
- Storage is as [z, x] or [z, y, x]
- These correspond to probes v, u in 2D, and w, v, u in 3D.
- Indexing as wave typically implies the member of a batch, and la the index into the wavelength array