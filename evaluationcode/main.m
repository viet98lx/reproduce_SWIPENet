classes = {'seacucumber', 'seaurchin', 'scallop', 'starfish'};
results=zeros(length(classes),1);

for i=1:length(classes)
    [rec,prec,ap]=Perclseval(classes{i}, 'true');
    results(i)=ap*100;
end
mAP=0;
for i=1:length(classes)
    disp([classes{i} 'AP: ' num2str(results(i))]);
    mAP= mAP+results(i);
end
mAP=mAP/length(classes);
disp(['mean AP: ' num2str(mAP)]);

