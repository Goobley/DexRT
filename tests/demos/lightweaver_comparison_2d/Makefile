test: VisualTest.png

VisualTest.png: J_10th.npy ../../build/output.nc
	python plot_output.py

J_10th.npy: simple_3_wavelength.py
	python $<

../../build/output.nc:
	../../lw_comparison_2d

