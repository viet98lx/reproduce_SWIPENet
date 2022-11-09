value=load('/scratch/deeplearn/lc408/plotROC/MyVOCcode/SSD512/scallop.mat');
% 'seacucumber','seaurchin', 'scallop'

x{1}=value.rec;
y{1}=value.prec;
% precision=load('Z:\My Documents\MATLAB\MyVOCcode_mouse\mouse\default_Faster_RCNN\precision.mat');
% recall=load('Z:\My Documents\MATLAB\MyVOCcode_mouse\mouse\default_Faster_RCNN\recall.mat');
% x{2}=recall.recall{1};
% y{2}=precision.precision{1};
% value=load('\\uol.le.ac.uk\root\staff\home\z\zj53\Desktop Files\MyVOCcode_mouse\mouse\Res50\head_curve.mat');
% x{2}=value.rec;
% y{2}=value.prec;
value=load('\\uol.le.ac.uk\root\staff\home\z\zj53\Desktop Files\MyVOCcode_mouse\mouse\Res101\head_curve.mat');
x{3}=value.rec;
y{3}=value.prec;
% value=load('\\uol.le.ac.uk\root\staff\home\z\zj53\Desktop Files\MyVOCcode_mouse\mouse\SSD300\head_curve.mat');
% x{4}=value.rec;
% y{4}=value.prec;
% value=load('\\uol.le.ac.uk\root\staff\home\z\zj53\Desktop Files\MyVOCcode_mouse\mouse\SSD512\head_curve.mat');
% x{5}=value.rec;
% y{5}=value.prec;
% value=load('\\uol.le.ac.uk\root\staff\home\z\zj53\Desktop Files\MyVOCcode_mouse\mouse\YOLO\head_curve.mat');
% x{6}=value.rec;
% y{6}=value.prec;

% methods = { 'VGG19+ours(AP:0.98)','Res50+fasterRCNN(AP:0.92)' ,'Res101+fasterRCNN(AP:0.98)','VGG16+SSD300(AP:0.87)','VGG16+SSD512(AP:0.98)','YOLO(AP:0.91)'};
methods = { 'SSD'};

% [~,test_labels]=max(target);
figure(5);

hold on
% title('Combination based ROC','FontSize',24)
set(gca,'FontSize',14);
xlabel('Recall','FontSize',14,'FontWeight','bold');
ylabel('Precision','FontSize',14,'FontWeight','bold');
set(gca, 'LineStyleOrder', {'-'});
for ii=1:length(methods)
    h1=plot(x{ii}(3:end), y{ii}(3:end),'LineWidth',2);
end
legend(methods,'FontSize',14);
ylim([0,1]);
saveas(h1,'roc.png');