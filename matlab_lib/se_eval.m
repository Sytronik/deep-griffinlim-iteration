function [pesq, stoi_val] = se_eval(clean, noisy, fs)

%[clean, clean_fs] = audioread(clean_name);
%[noisy, ~] = audioread(enhanced_name);

%fwsnr = fwsegsnr(clean, noisy, fs);

res = pesq_mex_vec(clean, noisy, fs);
pesq=mos2pesq(res);


stoi_val = stoi(double(clean), double(noisy), fs);

% clean = clean/max(abs(clean));
% noisy = noisy/max(abs(noisy));

% size(clean')
% size(noisy')
% snr = segsnr(clean', noisy', fs);
% plot(noisy - clean);
clc
end

