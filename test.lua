require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'xlua'
require 'csvigo'

train_file = 'train_32x32.t7'
test_file = 'test_32x32.t7'
out_file = 'predictions.csv'
trsize = 73257
tesize = 26032
loaded = torch.load(train_file,'ascii')
trainData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}
loaded = torch.load(test_file,'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return tesize end
}

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

print 'Load data TEST_OK'

--Load Trained model from file
print '==> loading trained model'
model = torch.load('model.net')

-- test over test data
print('==> testing on test set:')
Prediction = torch.zeros(tesize)
for t = 1,testData:size() do
    -- disp progress
    xlua.progress(t, testData:size())
    
    -- get new sample
	local input = testData.data[t]:double()
    local target = testData.labels[t]
    
    -- test sample
    local val, loc= torch.max(model:forward(input),1)
    -- print("\n" .. target .. "\n")
    if loc == 10 then
    	loc = 0
    end
    Prediction[t] = loc
end

--output to file
file = io.open(out_file, "w")
io.output(file)
io.write("Id,Prediction\n")
for i =1,testData:size() do
    io.write(tostring(i)..","..tostring(Prediction[i]).."\n")
    end
io.close(file)