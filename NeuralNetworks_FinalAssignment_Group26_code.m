load 'dataset_final_assignment.mat';

% grid size
max_x = 600;
max_y = 800;
end_time = size(data,1);
number_of_agents = size(data,2)/2;

% pre-process data (mirror all people over y-axis, !only if not already done!)
if data(1,2) > max_y/2
  for i = 1:end_time
    for a = 1:size(data(1,:),2)/2
      data(i,2*a) = max_y-data(i,2*a);
    end
  end
end

% relevant points (e.g. corners of buildings)
% NOTE: "max_y" operation was added, because image needs to be y-mirrorred
points = [546.8, max_y-478.0;
          507.5, max_y-330.6;
          240.6, max_y-218.6;
          184.7, max_y-331.3];

% sources of panic (e.g. shouting individual)
source = [542.0, max_y-439.0];

% solid lines that cannot be passed (buildings, fences, ...)
% NOTE: some lines have been made longer than represented in the data
lines = [321.2, max_y-314.5, 240.0, 300.0;          % was 321.2, max_y-314.5, 286.1, max_y-396.9;
         321.2, max_y-314.5, 275.5, max_y-292.0;
         383.0, max_y-336.4, 342.0, 300.0;          % was 383.0, max_y-336.4, 365.2, max_y-407.2
         383.0, max_y-336.4, 600.0, 358.0;          % was 383.0, max_y-336.4, 431.2, max_y-359.6
         385.3, max_y-321.6, 448.0, max_y-347.4;
         448.0, max_y-347.4, 449.0, max_y-313.9;
         449.0, max_y-313.9, 390.5, max_y-292.0;
         390.5, max_y-292.0, 385.3, max_y-321.6];

% calculate slopes and bases of solid lines
for l = 1:size(lines, 1)
    slopes(l) = (lines(l, 4) - lines(l, 2)) / (lines(l, 3) - lines(l, 1));
    bases(l) = lines(l, 2) - slopes(l) * lines(l, 1);
end

% save original data
dataOriginal=data;

% extra agents: 4 (left,right,top,bottom)
dataExtra=data;
delta=2;
 for a = 1:number_of_agents
    dataExtra = [dataExtra dataExtra(:,(2*a)-1)-delta dataExtra(:,(2*a)) dataExtra(:,(2*a)-1)+delta dataExtra(:,(2*a)) dataExtra(:,(2*a)-1) dataExtra(:,(2*a))-delta dataExtra(:,(2*a)-1) dataExtra(:,(2*a))+delta];
 end
 number_of_agents = number_of_agents*5;
 
% normalise data
% substract value of source
dataSource=dataExtra;
for a = 1:number_of_agents
    dataSource(:,(2*a)-1) = (dataSource(:,(2*a)-1)-source(1));
    dataSource(:,(2*a)) = (dataSource(:,(2*a))-source(2));
end

%maxSource_x=max_x-source(1);
%maxSource_y=max_y-source(2);
%minSource_x=-source(1);
%minSource_y=-source(2);

% cartesian2polar coordinates
data2pol=dataSource;
for a = 1:number_of_agents  
    x=data2pol(:,((2*a)-1));
    y=data2pol(:,2*a);
    [angel,length]=cart2pol(x,y);
    data2pol = [data2pol angel length];
end
data2pol=data2pol(:,351:700);

% split theta and rho
theta=data2pol(:,1:2:end);
rho=data2pol(:,2:2:end);

% normalize data
min_theta=min(min(theta));
max_theta=max(max(theta));
min_rho=min(max(rho));
max_rho=max(max(rho));
mean_theta=mean2(theta);
mean_rho=mean2(rho);
std_theta=std2(theta);
std_rho=std2(rho);

theta_norm=(theta-mean_theta)/std_theta;
rho_norm=(rho-mean_rho)/std_rho;

input_theta=theta_norm(1:46,:);
input_rho=rho_norm(1:46,:);

% not normalized
input_theta_old=theta(1:46,:);
input_rho_old=rho(1:46,:);
output_theta=theta_norm(2:47,:);
output_rho=rho_norm(2:47,:);
% not normalized
output_theta_old=theta(2:47,:);
output_rho_old=theta(2:47,:);

input_theta_split=[];
input_rho_split=[];
output_theta_split=[];
output_rho_split=[];
% not normalized
input_theta_old_split=[];
input_rho_old_split=[];
output_theta_old_split=[];
output_rho_old_split=[];

for a=1:175
    input_theta_split = [input_theta_split; input_theta(:,a)];
    input_rho_split = [input_rho_split; input_rho(:,a)];
    output_theta_split = [output_theta_split; output_theta(:,a)];
    output_rho_split = [output_rho_split; output_rho(:,a)];
    % not normalized
    input_theta_old_split = [input_theta_old_split; input_theta_old(:,a)];
    input_rho_old_split = [input_rho_old_split; input_rho_old(:,a)];
    output_theta_old_split = [output_theta_old_split; output_theta_old(:,a)];
    output_rho_old_split = [output_rho_old_split; output_rho_old(:,a)];
end

% time label
times=[];
for i=1:46
    %times=[times;i(ones(175,1),:)];
    times=repmat(tel,1,175);
end

input = [input_theta_split input_rho_split];
output = [output_theta_split output_rho_split];
% including time
input_time = [input times'];
input_old = [input_theta_old_split input_rho_old_split];
output_old = [output_theta_old_split output_rho_old_split];
input_notime = input_time(:,1:2);

% configure net
net = feedforwardnet([ 20 ],'trainlm');
net = configure(net,input_notime',output');
indices = cvpartition(input_notime(:,1)','k',10);
net2 = feedforwardnet([ 2 2 2 ], 'trainlm');
net2 = configure(net2,input_notime',output');
net3 = feedforwardnet([ 30 30 ], 'trainlm');
net3 = configure(net3,input_time',output');

perf=0;
for i = 1:indices.NumTestSets
    trIdx = indices.training(i);
    teIdx = indices.test(i);
    [net, tr] = trainscg(net,input_notime(trIdx,:)',output(trIdx,:)');
    [net2, tr] = trainscg(net2,input_notime(trIdx,:)',output(trIdx,:)');
    [net3, tr] = trainscg(net3,input_time(trIdx,:)',output(trIdx,:)');
    net_output = net(input_notime(teIdx,:)');
    net2_output = net2(input_notime(teIdx,:)');
    net3_output = net3(input_time(teIdx,:)');
    outputtest=output(teIdx,:)';
    perfExtra = perform(net,outputtest,net_output);
    perf = perf+perfExtra;
end
perf/indices.NumTestSets

% save teIdx for plotting with same input
save_teIdx=teIdx;
teIdx=save_teIdx;

output_denorm = [((output(:,1)*std_theta)+mean_theta) ((output(:,2)*std_rho)+mean_rho)];
[output_denorm_x, output_denorm_y] = pol2cart(output_denorm(:,1),output_denorm(:,2));
output_denorm_x_start = (output_denorm_x+source(1));
output_denorm_y_start = (output_denorm_y+source(2));

% for plotting de-normalize 
net_output_denorm = [((net_output(1,:)*std_theta)+mean_theta); ((net_output(2,:)*std_rho)+mean_rho)];
[net_output_denorm_x, net_output_denorm_y] = pol2cart(net_output_denorm(1,:),net_output_denorm(2,:));
net_output_denorm_x_start = (net_output_denorm_x+source(1));
net_output_denorm_y_start = (net_output_denorm_y+source(2));

% 2
net2_output_denorm = [((net2_output(1,:)*std_theta)+mean_theta); ((net2_output(2,:)*std_rho)+mean_rho)];
[net2_output_denorm_x, net2_output_denorm_y] = pol2cart(net2_output_denorm(1,:),net2_output_denorm(2,:));
net2_output_denorm_x_start = (net2_output_denorm_x+source(1));
net2_output_denorm_y_start = (net2_output_denorm_y+source(2));

% 3
net3_output_denorm = [((net3_output(1,:)*std_theta)+mean_theta); ((net3_output(2,:)*std_rho)+mean_rho)];
[net3_output_denorm_x, net3_output_denorm_y] = pol2cart(net3_output_denorm(1,:),net3_output_denorm(2,:));
net3_output_denorm_x_start = (net3_output_denorm_x+source(1));
net3_output_denorm_y_start = (net3_output_denorm_y+source(2));

output_denorm_x_start_part = output_denorm_x_start(teIdx,:);
output_denorm_y_start_part = output_denorm_y_start(teIdx,:);

%%%%
%Plot
%%%%
clf;
hold on;
grid on;
set(gcf, 'Position', [265 5 750 1000])
set(gca,'DataAspectRatio',[1 1 1]);
axis([0 max_x+1 0 max_y+1]);
    
  % draw lines
  for l = 1:size(lines, 1)
      line([lines(l,1) lines(l,3)],[lines(l,2) lines(l,4)],'Color',[0 0 0],'LineStyle','-');
  end

  % draw environmental objects (circle)
  radius = sqrt((386.9-374.3)^2+(208.9-257.2)^2);
  t=(0:50)*2*pi/50;
  x=radius*cos(t)+386.9;
  y=radius*sin(t)+max_y-208.9;
  plot(x,y,'Color',[0 0 0]);
 
  % draw relevant points
  for l = 1:size(points, 1)
        plot(points(l,1),points(l,2),'Color',[0 0 0],'Marker','.','MarkerSize',20);
  end

  % draw source
  plot(source(1),source(2),'Color',[1 0 0],'Marker','.','MarkerSize',20);
  % draw people
  for i=1:20:805 
    %plot(net_output_denorm_x_start(i),net_output_denorm_y_start(i),'Color',[0 0 1],'Marker','.','MarkerSize',7);
    %plot(net2_output_denorm_x_start(i),net2_output_denorm_y_start(i),'Color',[0 0 1],'Marker','.','MarkerSize',7);
    plot(net3_output_denorm_x_start(i),net3_output_denorm_y_start(i),'Color',[0 0 1],'Marker','.','MarkerSize',7);
    plot(output_denorm_x_start_part(i),output_denorm_y_start_part(i),'Color',[0.3 0.5 0],'Marker','x','MarkerSize',7);
  end