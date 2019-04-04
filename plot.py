
import h5py
import pickle
import logging
import numpy as np
import pylab as plt

from matplotlib.backends.backend_pdf import PdfPages

from taholog import misc
from taholog import sols_to_mods as s2m
from holog.fourierimaging import dft2_fast

def plot_phase_beam(beam_data_files_func, outplot, spws, refs):
    """
    Generates a plot showing the phase of the beam.
    """

    nx = 2
    ny = 5

    logger = logging.getLogger(__name__)

    fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
    grid = plt.GridSpec(ny, nx, wspace=0.1, hspace=0.3)

    ax = fig.add_subplot(111)
            
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # Set common labels
    ax.set_xlabel('Time sample')
    ax.set_ylabel('Phase (rad)')

    for s,spw in enumerate(spws):

        y = s//nx
        x = s%nx
  
        logger.info('s,x,y: {0}, {1}, {2}'.format(s, x, y))
 
        ax = fig.add_subplot(grid[y,x])
 
        for r,ref in enumerate(refs.split(',')):

            tf = beam_data_files_func(ref, spw)
            logger.info('Reading file: {0} .'.format(tf))

            fth5 = h5py.File(tf, 'r')
            ht = fth5.attrs
            beams = list(fth5.keys())

            # Load the central beam data.
            logger.info('Loading data, sigma and flag.')
            dt = np.array(fth5['/{0}/DATA'.format(beams[0])].get('data').value, dtype=np.complex64)
            st = np.array(fth5['/{0}/SIGMA'.format(beams[0])].get('sigma').value, dtype=np.complex64)
            ft = np.array(fth5['/{0}/FLAG'.format(beams[0])].get('flag').value, dtype=np.bool)

            # Apply flags to the data.
            dt = np.ma.masked_where(ft, dt)
            st = np.ma.masked_where(ft, st)

            # Load frequency axis.
            tgt_freq = np.array([fth5['/{0}/FREQ'.format(b)].get('freq').value for b in beams])

            # Compute the argument of the visibilities.
            phase = np.angle(dt[:,0,0])

            if r == 0:
                ax.set_title('{0:.2f} MHz'.format(np.mean(tgt_freq)*1e-6))

            if s == 0:

                ax.plot(phase, label='{0}'.format(ref))

            else:

                ax.plot(phase)

            ax.set_ylim(-np.pi, np.pi)

            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                           bottom=True, top=True, left=True, right=True)

            if x > 0:
                ax.yaxis.set_ticklabels([])

            if y < ny - 1:
                ax.xaxis.set_ticklabels([])

        if s == 0:
            ax.legend(loc=0, fancybox=True)

    #fig.tight_layout()
    plt.savefig(outplot, bbox_inches='tight')
    plt.close()

    logger.info('Finished plotting.')

def plot_report(output, solutions_file, uvhol_files_func, phase_ref_station='', 
                steps=['amplitude', 
                       'phase',
                       'phase_unwrapped',
                       'phase_referenced',
                       'delay',
                       'delay0',
                       'beam_observed',
                       'beam_model',
                       'beam_model_fit',
                       'beam_residuals',
                       'beam_residuals_fit',
                       'beam_dft',
                       'residuals_dft',
                       'residuals_fit_dft',
                       'phase_fit',
                       'plot_sols'],
                map_diameter_m=4000.0, ):
    """
    """

    logger = logging.getLogger(__name__)

    # Load solutions.
    sols = pickle.load(open(solutions_file, "rb"))
    
    ants = sorted(sols.keys())                       # Stations.
    nant = len(ants)                                 # Number of stations.
    spws = np.arange(0, len(sols[ants[0]]['freq']))  # Spectral windows.

    # Split behaviour between HBA and LBA because of "ears".
    if 'LBA' in ants[0]:
        antenna_type = 'LBA'
        ears = [0]
        ear_step = 1
        c_ = -1
    else:
        antenna_type = 'HBA'
        ears = [0,1]
        ear_step = 2
        c_ = -2

    # Maximum 25 stations per page.
    nx = 5
    ny = 5
    nx_b = 4
    ny_b = 3
    cm = plt.cm.cool

    pdfdoc = PdfPages(output)

    for ear in ears:
        
        if 'amplitude' in steps:
            
            logger.info('Plotting amplitude sols.')
            
            amp_max = 0
            amp_min = 0
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('Amplitude')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Gain amplitude')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                
                freq = sols[ant]['freq'][:]
                
                n_lines = sols[ant]['ph'].shape[0]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])
                
                ax.set_title(ant)
                
                for l in range(n_lines):
                    ax.errorbar(freq*1e-6, sols[ant]['amp'][l,0,:], yerr=sols[ant]['amp_err'][l,0,:], color=cm(color_idx[l]), ls='', marker='.')
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                amp_max = np.max((amp_max, np.max(sols[ant]['amp'])))
                amp_min = np.min((amp_min, np.min(sols[ant]['amp'])))
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
            for i,ant in enumerate(ants[ear::ear_step]):
            
                y = i//ny
                x = i%nx
                
                ax = fig.add_subplot(grid[y,x])
                
                ax.set_ylim(amp_min, amp_max)
            
            pdfdoc.savefig(fig)
            plt.close(fig)
        
        if 'phase' in steps:
            
            logger.info('Plotting phase sols.')
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('Phase')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Phase (rad)')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                        
                freq = sols[ant]['freq'][:]
                
                n_lines = sols[ant]['ph'].shape[0]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])

                ax.set_title(ant)
                
                for l in range(n_lines):
                    ax.plot(freq*1e-6, np.deg2rad(sols[ant]['ph'][l,0,:]), color=cm(color_idx[l]), ls='-', marker='.')
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
                ax.set_ylim(-np.pi*1.2, np.pi*1.2)
            
            pdfdoc.savefig(fig)
            plt.close(fig)
            
        if 'phase_unwrapped' in steps:
            
            logger.info('Plotting phase unwrapped sols.')
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('Phase unwrapped')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Phase (rad)')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                        
                freq = sols[ant]['freq'][:]
                
                n_lines = sols[ant]['ph_unwrap'].shape[0]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])

                ax.set_title(ant)
                
                for l in range(n_lines):
                    ax.plot(freq*1e-6, sols[ant]['ph_unwrap'][l,0,:], color=cm(color_idx[l]), ls='-', marker='.')
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
                ax.set_ylim(-np.pi*3, np.pi*3)
            
            pdfdoc.savefig(fig)
            plt.close(fig)
        
        if 'phase_referenced' in steps:
            
            logger.info('Plotting phase referenced sols.')
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('Phase referenced')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Phase (rad)')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                        
                freq = sols[ant]['freq'][:]
                
                n_lines = sols[ant]['ph_ref'].shape[0]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])

                ax.set_title(ant)
                
                for l in range(n_lines):
                    phases = sols[ant]['ph_ref'][l,0,:]
                    #phases = wrap_phase(sols[ant]['ph_ref'][l,0,:])
                    #phases = np.unwrap(sols[ant]['ph_ref'][l,0,:], axis=-1)
                    ax.plot(freq*1e-6, phases, color=cm(color_idx[l]), ls='-', marker='.')
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
                ax.set_ylim(-np.pi*2., np.pi*2.)
            
            pdfdoc.savefig(fig)
            plt.close(fig)
            
        if 'delay' in steps:
            
            logger.info('Plotting delays.')
            
            tau_max = 0
            tau_min = 0
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('Delay')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Time slot')
            ax.set_ylabel('Time delay (ns)')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                        
                time = np.arange(sols[ant]['tau'].shape[0])
                
                n_lines = sols[ant]['tau'].shape[1]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])

                ax.set_title(ant)
                
                for l in range(n_lines):
                    ax.errorbar(time, sols[ant]['tau'][:,l]*1e9, yerr=sols[ant]['tau_err'][:,l]*1e9, color=cm(color_idx[l]), ls='', marker='.')
                
                tau_max = np.max((tau_max, np.max(sols[ant]['tau'][:,l]*1e9)))
                tau_min = np.min((tau_min, np.min(sols[ant]['tau'][:,l]*1e9)))
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                
                ax = fig.add_subplot(grid[y,x])
                
                ax.set_ylim(tau_min, tau_max)
                            
            pdfdoc.savefig(fig)
            plt.close(fig)
            
        if 'delay0' in steps:
            
            logger.info('Plotting 0 Hz phase offsets.')
            
            tau0_max = 0
            tau0_min = 0
            
            fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
            grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
            
            fig.suptitle('0 Hz phase offset')
            
            ax = fig.add_subplot(111)
            
            # Turn off axis lines and ticks of the big subplot
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

            # Set common labels
            ax.set_xlabel('Time slot')
            ax.set_ylabel('0 Hz phase offset (rad)')
            
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                        
                time = np.arange(sols[ant]['tau0'].shape[0])
                
                n_lines = sols[ant]['tau0'].shape[1]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])

                ax.set_title(ant)
                
                for l in range(n_lines):
                    ax.errorbar(time, sols[ant]['tau0'][:,l], yerr=sols[ant]['tau0_err'][:,l], color=cm(color_idx[l]), ls='', marker='.')
                
                tau0_max = np.max((tau0_max, np.max(sols[ant]['tau0'][:,l])))
                tau0_min = np.min((tau0_min, np.min(sols[ant]['tau0'][:,l])))
                
                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                               bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
            for i,ant in enumerate(ants[ear::ear_step]):
                
                y = i//ny
                x = i%nx
                
                ax = fig.add_subplot(grid[y,x])
                
                ax.set_ylim(tau0_min, tau0_max)
                            
            pdfdoc.savefig(fig)
            plt.close(fig)
            
        # End of iteration by ears.

    # Start 2D beam related plots.
    
    if 'beam' in ','.join(steps):
        # Generate beam models.
        s2m_out = s2m.make_model(spws, sols, uvhol_files_func, phase_ref_station)
        
        laxis = s2m_out['laxis']
        maxis = s2m_out['maxis']
        extent = np.array([laxis.min(), laxis.max(), maxis.min(), maxis.max()]) * 1e3
        station_pqr = s2m_out['station_pqr']        

    if 'beam_observed' in steps:
        
        logger.info('Plotting observed beam.')
        
        # Plot the observed beam.
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Observed beam')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('l (mrad)')
        ax.set_ylabel('m (mrad)')
        
        for i,spw in enumerate(spws[:-1]):
                                
            ax = fig.add_subplot('33{0}'.format(i+1))

            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            im = ax.imshow(abs(s2m_out['obs_beam'][i]), origin='lower', extent=extent)
            plt.colorbar(im, ax=ax)
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'beam_model' in steps:
        # Plot the observed beam.
        logger.info('Plotting modeled beam.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Model beam')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('l (mrad)')
        ax.set_ylabel('m (mrad)')
        
        for i,spw in enumerate(spws[:-1]):
                                
            ax = fig.add_subplot('33{0}'.format(i+1))

            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            im = ax.imshow(abs(s2m_out['mod_beam'][i]), origin='lower', extent=extent)
            plt.colorbar(im, ax=ax)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'beam_model_fit' in steps:
        # Plot the observed beam.
        logger.info('Plotting modeled beam from fits to phase solutions.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Model beam fit')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('l (mrad)')
        ax.set_ylabel('m (mrad)')
        
        for i,spw in enumerate(spws[:-1]):
                                
            ax = fig.add_subplot('33{0}'.format(i+1))

            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            im = ax.imshow(abs(s2m_out['mod_beam_fit'][i]), origin='lower', extent=extent)
            plt.colorbar(im, ax=ax)
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'beam_residuals' in steps:
        # Plot observed beam minus modeled beam.
        logger.info('Plotting observed beam residuals.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Observed beam - model beam')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('l (mrad)')
        ax.set_ylabel('m (mrad)')
        
        for i,spw in enumerate(spws[:-1]):
                                
            ax = fig.add_subplot('33{0}'.format(i+1))

            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            im = ax.imshow(abs(s2m_out['obs_beam'][i] - s2m_out['mod_beam'][i]), origin='lower', extent=extent)
            plt.colorbar(im)
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'beam_residuals_fit' in steps:
        # Plot observed beam minus modeled beam.
        logger.info('Plotting observed beam residuals using fits to the phases.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Observed beam - model beam fit')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('l (mrad)')
        ax.set_ylabel('m (mrad)')
        
        for i,spw in enumerate(spws[:-1]):
                                
            ax = fig.add_subplot('33{0}'.format(i+1))

            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            im = ax.imshow(abs(s2m_out['obs_beam'][i] - s2m_out['mod_beam_fit'][i]), origin='lower', extent=extent)
            plt.colorbar(im)
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'beam_dft' in steps:
        # Plot the discrete Fourier transform of the observed beam.
        logger.info('Plotting observed beam DFT.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Observed beam DFT')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('p (m)')
        ax.set_ylabel('q (m)')
        
        for i,spw in enumerate(spws[:-1]):
            
            hd_ai = s2m_out['holog_data'][i].complex_aperture_dft(map_diameter_m=map_diameter_m, over_sample_factor=5, fourier_sign=-1)
                                
            ax = fig.add_subplot('33{0}'.format(i+1))
            
            grid_ = s2m_out['holog_data'][i].ft_grid(map_diameter_m, 5, -1)
            res_hd_dft = dft2_fast(grid_, v=s2m_out['holog_data'][i].vis)
            
            ax = hd_ai.image().plot(function=abs, interpolation='nearest', cbar=True, axes=ax)
            ax.set_title('')
            
            im = ax.images[0]
            im.set_data(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum()))
            #im.set_data(abs((res_hd_ai.image_data))*len(hd[0].vis)**2./abs(np.sum(hd[0].vis)))
            im.set_clim(vmin=0, vmax=np.max(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum())))
            ax.scatter(-s2m_out['station_pqr'][:,0], -s2m_out['station_pqr'][:,1], s=20, facecolor='none', edgecolor='white', alpha=0.5)
            
            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'residuals_dft' in steps:
        # Plot the discrete Fourier transform of the observed beam minus modeled beam.
        logger.info('Plotting DFT of the residuals.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Residuals DFT (%)')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('p (m)')
        ax.set_ylabel('q (m)')
        
        for i,spw in enumerate(spws[:-1]):
            
            obs_hd_ai = s2m_out['holog_data'][i].complex_aperture_dft(map_diameter_m=map_diameter_m, over_sample_factor=5, fourier_sign=-1)
            mod_hd_ai = s2m_out['model_hologdata'][i].complex_aperture_dft(map_diameter_m=map_diameter_m, over_sample_factor=5, fourier_sign=-1)
            res_hd_ai = s2m_out['residual_hologdata'][i].complex_aperture_dft(map_diameter_m=map_diameter_m, over_sample_factor=5, fourier_sign=-1)
            
            ax = fig.add_subplot('33{0}'.format(i+1))
            
            grid_ = s2m_out['residual_hologdata'][i].ft_grid(map_diameter_m, 5, -1)
            res_hd_dft = dft2_fast(grid_, v=s2m_out['residual_hologdata'][i].vis)

            ax = res_hd_ai.image().plot(function=abs,interpolation='nearest', cbar=True)
            im = ax.images[0]
            im.set_data(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum()))
            #im.set_data(abs((res_hd_ai.image_data))*len(hd[0].vis)**2./abs(np.sum(hd[0].vis)))
            im.set_clim(vmin=0, vmax=np.max(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum())))
            
            
            #ax = mod_hd_ai.image().plot(function=abs,interpolation='nearest', cbar=True, axes=ax)
            #im = ax.images[0]
            #im.set_data(abs((obs_hd_ai.image_data - mod_hd_ai.image_data))/abs(obs_hd_ai.image_data))
            #im.set_clim(vmin=0, vmax=1)
            #ax.set_
            ax.set_title('')
            ax.scatter(-station_pqr[:,0], -station_pqr[:,1], s=20, facecolor='none', edgecolor='white', alpha=0.5)
            
            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'residuals_fit_dft' in steps:
        # Plot the discrete Fourier transform of the observed beam minus modeled beam.
        logger.info('Plotting DFT of the residuals using the fits to the phases.')
        
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        #grid = plt.GridSpec(nx_b, ny_b, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Residuals Fit DFT (%)')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('p (m)')
        ax.set_ylabel('q (m)')
        
        for i,spw in enumerate(spws[:-1]):
            
            res_hd_ai = s2m_out['residual_fit_hologdata'][i].complex_aperture_dft(map_diameter_m=map_diameter_m, over_sample_factor=5, fourier_sign=-1)
            
            ax = fig.add_subplot('33{0}'.format(i+1))
            
            grid_ = s2m_out['residual_hologdata'][i].ft_grid(map_diameter_m, 5, -1)
            res_hd_dft = dft2_fast(grid_, v=s2m_out['residual_fit_hologdata'][i].vis)

            ax = res_hd_ai.image().plot(function=abs,interpolation='nearest', cbar=True)
            im = ax.images[0]
            im.set_data(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum()))
            im.set_clim(vmin=0, vmax=np.max(abs(res_hd_dft/s2m_out['holog_data'][i].vis.sum())))
            
            ax.set_title('')
            ax.scatter(-station_pqr[:,0], -station_pqr[:,1], s=20, facecolor='none', edgecolor='white', alpha=0.5)
            
            ax.set_title('{0:.2f} MHz'.format(sols[ants[0]]['freq'][i]*1e-6))
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            ax.minorticks_on()
            ax.tick_params('both', direction='in', which='both',
                            bottom=True, top=True, left=True, right=True)
            
            if i not in [0, 3, 6]:
                ax.yaxis.set_ticklabels([])
            
            if i not in [6, 7, 8]:
                ax.xaxis.set_ticklabels([])
                        
        pdfdoc.savefig(fig)
        plt.close(fig)
        
    if 'phase_fit' in steps:
        
        logger.info('Plotting fits to the phases.')
            
        fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
        grid = plt.GridSpec(nx, ny, wspace=0.1, hspace=0.3)
        
        fig.suptitle('Phase fit')
        
        ax = fig.add_subplot(111)
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Set common labels
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Phase (rad)')
        
        for ear in ears:

            for i,ant in enumerate(ants[ear::ear_step]):
            
                y = i//ny
                x = i%nx
                        
                freq = sols[ant]['freq'][:]
                
                n_lines = sols[ant]['ph_ref'].shape[0]
                color_idx = np.linspace(0, 1, n_lines)
                
                ax = fig.add_subplot(grid[y,x])
                
                ax.set_title(ant[:c_])
                
                if ear == 0: 
                    l0, = ax.plot(freq*1e-6, s2m_out['phs_sol_fit'][:,ear+i*ear_step], color='g', ls='-', marker='.')
                    l1, = ax.plot(freq*1e-6, np.deg2rad(s2m_out['phs_sol'][:,ear+i*ear_step]), color='g', ls=':', marker='.')
                else:
                    l2, = ax.plot(freq*1e-6, s2m_out['phs_sol_fit'][:,ear+i*ear_step], color='b', ls='-', marker='.')
                    l3, = ax.plot(freq*1e-6, np.deg2rad(s2m_out['phs_sol'][:,ear+i*ear_step]), color='b', ls=':', marker='.')
                
                if x == 0 and y == 0:

                    ax.legend([l0,l1], ['fit 0', 'obs 0'], loc=0, prop={'size': 6})

                ax.minorticks_on()
                ax.tick_params('both', direction='in', which='both',
                                bottom=True, top=True, left=True, right=True)
                
                if x > 0:
                    ax.yaxis.set_ticklabels([])
                
                if y < ny - 1:
                    ax.xaxis.set_ticklabels([])
                    
                ax.set_ylim(-np.pi*2., np.pi*2.)
        
        pdfdoc.savefig(fig)
        plt.close(fig)

    # End of beam related plots.
    
    if 'plot_sols' in steps:
        
        logger.info('Plotting solutions: time delays and phase offsets.')        
        fig = plot_sols(sols)
        pdfdoc.savefig(fig)
        plt.close(fig)
 
    pdfdoc.close()

    logger.info('Done plotting {0}'.format(output))

def plot_sols(sols):
    """
    Plots the time delays and phase offsets for the stations.
    """

    n_col = 5
    cols = 4

    ants = sorted(sols.keys())                      # Stations.
    nant = len(ants)                                 # Number of stations.
    spws = len(sols[ants[0]]['freq'])               # Spectral windows.

    tau = np.zeros((nant), dtype=np.float32)
    dtau = np.zeros((nant), dtype=np.float32)
    tau0 = np.zeros((nant), dtype=np.float32)
    dtau0 = np.zeros((nant), dtype=np.float32)

    for i,station in enumerate(ants):

        tau[i] = sols[station]['tau'][0,0]*1e9  # ns
        dtau[i] = sols[station]['tau_err'][0,0]*1e9
        tau0[i] = sols[station]['tau0'][0,0]
        dtau0[i] = sols[station]['tau0_err'][0,0]

    tau_max = np.max(tau) + dtau[np.argmax(tau)]
    tau_min = np.min(tau) - dtau[np.argmax(tau)]
    tau0_max = np.max(tau0) + dtau0[np.argmax(tau0)]
    tau0_min = np.min(tau0) - dtau0[np.argmax(tau0)]

    # Catch invalid values.
    tau = np.ma.masked_invalid(tau)
    tau0 = np.ma.masked_invalid(tau0)
    dtau = np.ma.masked_invalid(dtau)
    dtau0 = np.ma.masked_invalid(dtau0)

    # Axis to evaluate the Gaussians.
    tau_axis = np.arange(tau_min, tau_max, 0.05)
    tau0_axis = np.arange(tau0_min, tau0_max, 0.05)

    # Build Gaussians.
    tau_gau = np.zeros((len(ants), len(tau_axis)), dtype=np.float)
    tau0_gau = np.zeros((len(ants), len(tau0_axis)), dtype=np.float)

    for i,station in enumerate(ants):
        
        if station == sols[station]['ph_ref_station']:
            continue
         
        tau_gau[i] = misc.gaussian(tau_axis, dtau.filled(tau.std())[i], tau[i], 1)/misc.gauss_area(1, dtau.filled(tau.std())[i])
        tau0_gau[i] = misc.gaussian(tau0_axis, dtau0.filled(tau0.std())[i], tau0[i], 1)/misc.gauss_area(1, dtau0.filled(tau0.std())[i])

    fig = plt.figure(frameon=False, dpi=150, figsize=(6,6))
    
    grid = plt.GridSpec(2, n_col, wspace=0.1, hspace=0.1)
    
    ax = fig.add_subplot(grid[0,0:cols])
    
    ax.errorbar(np.arange(nant), tau, yerr=dtau.filled(tau.std()), c='green',  fmt='.', ls='')
    
    ax.set_xticks(range(0,nant))
    ax.set_xticklabels('')
    
    ax.set_ylabel(r'$\tau$ (ns)')
    
    
    ax.minorticks_on()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', direction='in', which='minor',
                    bottom=False, top=False, left=True, right=True)
    
    ax.set_ylim(tau_min, tau_max)
    
    ax = fig.add_subplot(grid[1,0:cols])
    
    ax.errorbar(np.arange(nant), tau0, yerr=dtau0.filled(tau0.std()), c='green',  fmt='.', ls='')
    
    ax.set_xticks(range(0,nant))
    x_labels = [ant if i%2==0 else '' for i,ant in enumerate(ants)]
    ax.set_xticklabels(x_labels, rotation=45, size=5, ha='right')
    
    ax.set_ylabel(r'$\phi_{0}$ (rad)')
    
    ax.set_xlabel('Station')
    
    ax.minorticks_on()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', direction='in', which='minor',
                    bottom=False, top=False, left=True, right=True)
    
    ax.set_ylim(tau0_min, tau0_max)
    
    ax = fig.add_subplot(grid[0,cols])
    
    ax.plot(tau_gau.sum(axis=0)/(len(ants) - 1), tau_axis, c='green', ls='-')
    
    ax.minorticks_on()
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    
    ax.set_ylim(tau_min, tau_max)
    
    ax = fig.add_subplot(grid[1,cols])
    
    ax.plot(tau0_gau.sum(axis=0)/(len(ants) - 1), tau0_axis, c='green', ls='-')
    
    ax.minorticks_on()
    ax.yaxis.tick_right()
    ax.tick_params('both', direction='in', which='both',
                    bottom=True, top=True, left=True, right=True)
    
    ax.set_ylim(tau0_min, tau0_max)

    return fig
