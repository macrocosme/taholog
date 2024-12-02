taholog
=======

Scripts for tied-array holography.

Author(s)
---------
Original work by P. Salas et al. 2020. *Astronomy & Astrophysics*. `635, A207 <https://www.aanda.org/articles/aa/full_html/2020/03/aa35670-19/aa35670-19.html>`_. 
Code refactored, modularized and optimized by `@macrocosme <https://github.com/macrocosme>`_.

Usage example
-------------

In a shell: 

.. code-block:: bash

  > python main.py --parallel \\
                   --ncpus 16 \\
                   --no-use_numba \\
                   --no-use_gpu \\
                   --use_pyfftw \\ 
                   --to_disk \\
                   --verbose \\
                   --target_id "<LOFAR TARGET OBS ID>" \\
                   --reference_ids "LOFAR REFERENCE OBS ID(S)" \\
                   --input_dir "</PATH/TO/INPUT_DATA>" \\
                   --output_dir "</PATH/TO/OUTPUT_DATA>" >> ../<LOG FILE NAME>.out 2>&1

For more:
``> python main.py --help``

Setup
------

Refer to ``setup.pdf`` for details. 
