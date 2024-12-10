def gp_model(data_dict):
    with pm.Model() as model:

        u = pm.Uniform("u", 0, 1)
        mu = 90 - tt.arccos(u) * 180 / np.pi
        pm.Deterministic("gp.mu", mu)
        sigma = pm.Uniform("gp.sigma", 1.0, 20.0)

        sp_model = StarryProcess(
            mu = mu,
            sigma = sigma,
            c = pm.Uniform("gp.c", 0.0, 1., testval=truths['gp.c']),
            r = pm.Uniform("gp.r", 10.0, 30.0),
            n = pm.Uniform("gp.n", 1.0, 30.0, testval=5),
            ydeg = 6,
        )

        planet_model = starry.kepler.Secondary(starry.Map(0,0), 
                t0=truths['planet.t0'], 
                r=truths['planet.r'], 
                m=truths['planet.m'],
                porb=truths['planet.porb'], 
                ecc=truths['planet.ecc'], 
                Omega=truths['planet.Omega'], 
                inc=truths['planet.inc'])

        for kic_id, star_data in data_dict.items():
            # log_prot = pm.Uniform(f'logp_{kic_id}',tt.log(truths['star.prot']) + np.log1p(0.2), 
            #                       tt.log(truths['star.prot']) + np.log1p(prot_frac_bounds))
            # prot = pm.Deterministic(f"prot_{kic_id}",tt.exp(log_prot))
            prot = pm.Uniform(f"prot_{kic_id}", 0.1, 40, testval=20)

            map_model = starry.Map(ydeg=6)
            cosi = pm.Uniform(f"cosi_{kic_id}", -1, 1, testval=tt.cos(truths['star.inc']*np.pi/180))
            inc = pm.Deterministic(f"inc_{kic_id}", 180.0/np.pi*tt.arccos(cosi))
            map_model.inc = inc

            star_model = starry.Primary(map_model, 
                r=1, 
                m=1, 
                prot=prot)

            sys_model = starry.System(star_model, planet_model)

            for chunk in range(data_dict[kic_id]['chunks'].shape[0]):
                ssp_model = StarryStarryProcess(sys_model, sp_model)

                t_model = data_dict[kic_id]['chunks'][chunk][0]
                flux_vals = data_dict[kic_id]['chunks'][chunk][1]
                err_vals = data_dict[kic_id]['chunks'][chunk][2]

                ssp_model.compute(t_model, flux_vals, err_vals)

                pm.Potential(f'marginal_likelihood_{kic_id}_{chunk}', ssp_model.marginal_likelihood(t_model, flux_vals, err_vals))
                
                pm.Deterministic(f'flux_model_{kic_id}_{chunk}', tt.dot(ssp_model.design_matrix, ssp_model.sample_ylm_conditional(t_model, flux_vals, err_vals)[0]))
                pm.Deterministic(f'max_map_{kic_id}_{chunk}', ssp_model._a)
            

    return model