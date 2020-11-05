# Build Doxygen docs
import os, subprocess
if not os.path.exists('_build'):
    os.makedirs('_build')
if not os.path.exists('_build/doxygen'):
    os.makedirs('_build/doxygen')
subprocess.call('cd _build/doxygen; doxygen ../../Doxyfile', shell=True)
