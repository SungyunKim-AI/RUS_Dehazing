setDir = 'D:\data\RESIDE_beta\train\clear';
imds = imageDatastore(setDir, 'FileExtensions',{'.jpg'});

model = fitniqe(imds);

pop_mu = model.Mean;
pop_cov = model.Covariance;

save('niqe_image_params.mat', 'pop_mu', '--append');
save('niqe_image_params.mat', 'pop_cov', '--append');

save('niqe_image_params.mat', 'pop_mu', 'pop_cov');