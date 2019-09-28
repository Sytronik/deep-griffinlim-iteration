function fwseg_dist= comp_fwseg(cleanFile, enhancedFile);

% ----------------------------------------------------------------------
%      Frequency weighted SNRseg Objective Speech Quality Measure
%
%   This function implements the frequency-weighted SNRseg measure [1]
%   using a different weighting function, the clean spectrum.
%
%   Usage:  fwSNRseg=comp_fwseg(cleanFile.wav, enhancedFile.wav)
%           
%         cleanFile.wav - clean input file in .wav format
%         enhancedFile  - enhanced output file in .wav format
%         fwSNRseg      - computed frequency weighted SNRseg in dB
% 
%         Note that large numbers of fwSNRseg are better.
%
%  Example call:  fwSNRseg =comp_fwseg('sp04.wav','enhanced.wav')
%
%  
%  References:
%   [1]  Tribolet, J., Noll, P., McDermott, B., and Crochiere, R. E. (1978).
%        A study of complexity and quality of speech waveform coders. Proc. 
%        IEEE Int. Conf. Acoust. , Speech, Signal Processing, 586-590.
%
%   Author: Philipos C. Loizou 
%  (critical-band filtering routines were written by Bryan Pellom & John Hansen)
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
% ----------------------------------------------------------------------

if nargin~=2
    fprintf('USAGE: fwSNRseg=comp_fwseg(cleanFile.wav, enhancedFile.wav)\n');
    fprintf('For more help, type: help comp_fwseg\n\n');
    return;
end


[data1, Srate1, Nbits1]= wavread(cleanFile);
[data2, Srate2, Nbits2]= wavread(enhancedFile);
if ( Srate1~= Srate2) | ( Nbits1~= Nbits2)
    error( 'The two files do not match!\n');
end

len= min( length( data1), length( data2));
data1= data1( 1: len)+eps;
data2= data2( 1: len)+eps;

wss_dist_vec= fwseg( data1, data2,Srate1);

fwseg_dist=mean(wss_dist_vec);

