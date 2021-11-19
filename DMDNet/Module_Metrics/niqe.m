setDir = fullfile(toolboxdir('images'), 'imdata');
imds = imageDatastore(setDir, 'FileExtensions',{'.jpg'});

T = imds.countEachLabel();

model = fitniqe(imds);
