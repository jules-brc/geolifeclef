from environmental_raster_glc import PatchExtractor
from environmental_raster_glc import raster_metadata

extractor = PatchExtractor('../data/rasters_GLC19', size=1, verbose=True)

#extractor.append('chbio_1')

for channel in raster_metadata.keys():
    if not channel in {'alti','clc','proxi_eau_fast'}:
        extractor.append(channel)

print(f'Dimension of the extractor: {len(extractor)}')
print(f'Dimension of a point: {extractor[43.61, 3.88].shape}')
print(f'Environmental tensor at (43.61,3.88) (Montpellier) :\n{extractor[43.61, 3.88]}')
