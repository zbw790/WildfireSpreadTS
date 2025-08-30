@echo off
cd /d "E:\github\WildfireSpreadTS"
echo Starting HDF5 conversion...
echo Data directory: data
echo Target directory: data\processed
echo.
python src\preprocess\CreateHDF5Dataset.py --data_dir data --target_dir data\processed
echo.
echo Conversion completed!
pause 