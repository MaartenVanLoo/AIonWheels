v = linspace(-1,10,10)' %m/s actual speed
v = abs(v)
v = max(v,4)
v1=15   % ego car target speed
v2=0    % front vehicle speed
safe_distance = 4 + 2 * v

distance = repmat(linspace(0,80,100),length(v2),1)
x= (distance - safe_distance)./(v)
alpha = 1./(1+exp(-x))*2-1
target = ...
    (alpha .* v1 + (1-alpha) .*v2) %.* (distance >= safe_distance) + ...
    %(alpha .* v2 + (1-alpha) .*v2) .* (distance < safe_distance)
plot(distance',target')
hold on
plot(distance,repmat(v1,1,numel(distance)))
plot(distance,repmat(v2,1,numel(distance)))
hold off