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
- [ ] Final formal solution (interactive?)
- [x] Non-LTE iteration (@5a68d12)
    - [ ] Add line profile normalisation in final cascade (`wphi`)
- [ ] Boundary conditions
    - [ ] Interpolated Prom style
        - Moved to fully 3D treatment -- looks mostly correct. Need to check ray integration in 3D -- may have to simplify to 2D given the integration method used currently
    - [ ] Periodic
        - Lagged sampling from previous cascade 0? -- very diffusive
        - Ping-pong full cascade stack and do a directional lookup/traversal
            - Could do this in a separate alg and build a BC from this
        - Hauschildt & Baron style multiple wrap-around?
    - Make generic via templating struct passed down to raymarcher -- each BC
    has a `sample` function so dispatch is fully static (there won't be many
    types so use a switch to template dispatch)
- [ ] Multiple atoms - basically splat everything SOA a la MULTI and like is done for the one atom here.
- [ ] PRD
- [ ] Sparse VDB-like grid


Ideas
=====

- [ ] Move LTE populations to be partition function based (allows for calculation of only the levels of interest)
- [x] Use magma to abstract batched LU solved for populations
    - [ ] Needs non-GPU alternative
- [ ] Store rays per cascade pre-averaged (groups of 4) - should reduce bandwidth and storage requirements, but requires atomic operation
- [x] Avoid local memory in dda ray traversal -- seems to be causing stalls
    - Implemented fully register-based approach using the switch/template method from nanovdb - big perf difference.
- [ ] Full per-wavelength active set treatment. GPU benefits a lot more from this


Notes
=====

Formalising coordinate system
------------------------------
- z up, x default perpendicular axis, y as per right-hand rule
- Storage is as [z, x] or [z, y, x]
- These correspond to probes v, u in 2D, and w, v, u in 3D.