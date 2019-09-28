function snr = fwsegsnr(clean_speech, processed_speech, fs)
    [len, ~] = size(clean_speech);
    if len==1
        clean_speech = clean_speech.';
        processed_speech = processed_speech.';
    end
    snr = mean(fwseg(clean_speech, processed_speech, fs));
end