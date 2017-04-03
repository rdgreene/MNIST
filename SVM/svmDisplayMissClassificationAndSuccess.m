
figure;
res = tTest ~= labelsOut;
resIndx = find(res);
size(resIndx)
chooseTen = resIndx([30:41]);
choosenPic = xTrain(chooseTen,:);
display_network(choosenPic')

figure;
res = tTest == labelsOut;
resIndx = find(res);
size(resIndx)
chooseTen = resIndx([440:451]);
choosenPic = xTrain(chooseTen,:);
display_network(choosenPic')