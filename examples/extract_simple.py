from environmental_raster_glc import PatchExtractor

extractor = PatchExtractor('../data/rasters_GLC19', size=64, verbose=True)

extractor.append('chbio_1')

# extractor.add_all()

print(len(extractor))
print(extractor[43.61, 3.88].shape)
print(extractor[43.61, 3.88])

# extractor.plot((43.61, 3.88))
