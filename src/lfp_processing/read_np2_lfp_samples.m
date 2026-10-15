function dataI16 = read_np2_lfp_samples(fileInfo, sampleStart, nSamples, nChannels)
% READ_NP2_LFP_SAMPLES - Read int16 frames from concatenated NP2 LFP files
%
% Variables:
%   fileInfo    - Struct array from find_np2_lfp_files (path, nSamples)
%   sampleStart - 0-based sample index into the stitched stream
%   nSamples    - Number of samples (time points) to read
%   nChannels   - Channel count in each frame
%
% Goal:
%   Return an nChannels x nSamples int16 matrix spanning file boundaries.
%   Values are the ADC codes preserved by extract_lfp_np2.sh.

if nargin < 4 || isempty(nChannels)
    nChannels = np2_lfp_constants().nChannels;
end
if nSamples <= 0
    dataI16 = zeros(nChannels, 0, 'int16');
    return
end

nFiles = numel(fileInfo);
fileNSamples = [fileInfo.nSamples];
cumSamples = cumsum(fileNSamples);
fileStarts = [0, cumSamples(1:end-1)];
nTotal = cumSamples(end);

if sampleStart < 0 || sampleStart + nSamples > nTotal
    error('read_np2_lfp_samples:OutOfRange', ...
        'Requested samples [%d, %d) outside stitched range [0, %d).', ...
        sampleStart, sampleStart + nSamples, nTotal);
end

dataI16 = zeros(nChannels, nSamples, 'int16');
nFilled = 0;
readPos = sampleStart;

while nFilled < nSamples
    fileIdx = find(readPos >= fileStarts & readPos < cumSamples, 1, 'first');
    localStart = readPos - fileStarts(fileIdx);
    nTake = min(fileNSamples(fileIdx) - localStart, nSamples - nFilled);

    fid = fopen(fileInfo(fileIdx).path, 'r', 'ieee-le');
    if fid < 0
        error('read_np2_lfp_samples:OpenFailed', 'Could not open %s', fileInfo(fileIdx).path);
    end
    closer = onCleanup(@() fclose(fid));
    fseek(fid, localStart * nChannels * 2, 'bof');
    chunk = fread(fid, [nChannels, nTake], 'int16=>int16');
    clear closer
    if size(chunk, 2) ~= nTake
        error('read_np2_lfp_samples:ShortRead', ...
            'Expected %d samples from %s, got %d.', nTake, fileInfo(fileIdx).name, size(chunk, 2));
    end

    dataI16(:, nFilled + (1:nTake)) = chunk;
    nFilled = nFilled + nTake;
    readPos = readPos + nTake;
end
end
