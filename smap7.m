##S mapping
pkg load statistics

#load 'logistic.dat' #load up and pre-process a timeseries of data from the logisic map to use as an example
#as it is chaotic.
#data=flipud(logistic); #flip so that most recent at top, so not going backwards.

load 'tinkerbell.dat'
data=flipud(tinkerbell);

latest = 90
earliest = 100
x=data (latest:earliest,:);





#may need to then one minus the data if doing
#one step projection, or Tp minus if doing Tp steps.
m=length(x); #define m and d, theta and Tp (last 3 will be in function when done)
d=2;
Tp=1
theta=2

#function smap = map(x,d,Tp,theta,y)
#always rescale so Xt(0) = 1
x = x.+(1-(x(m,:)));
#x=(x(1:m-1)).-(x(2:m)) # '1 minusing' the data reduced length by 1 (or Tp)
#x=(x(1:m-Tp)).-(x(Tp+1:m))
#redefine m: now x i Tp shorter.
#m = length(x);

#embed the time series:
#embed without using a loop
#for repelems need to make a vector specifying the index in row 1 and the no of repeats in r2

#r = 1:m
#r2 = [zeros(1,d-1),[ones(1,(m-(d-1)))]]
#R=[r;r2]

#then use repelems to create the embedding vector
#emvect = repelems(x',R)
#add this to x
#embedded = [x(1:m-(d-1));emvect']'

#should actually be done to allow changes in the embedding dimension.  

#embed the time series:
m = length(x);
embedded= ones((m-d),1); #presize a matrix
for i= 1:d;
col= x([i:(m-(d-(i-1)))],1);
embedded= [embedded, col];
endfor
# this embedded has the original column of 1's, so chop them off:
embedded;
embedded= embedded(:,2:(d+1));


#embedded is now a matrix of d+1 columns and n-d-1 rows. 
[n,dim]= size(embedded);# get size of the embedded time series matrix

#calculate distances between all points to get distbar
#can do this using pdist (statistics package)
#need to load statistics if not already done

distbar =mean(pdist(embedded))  

#now need the distance between latest simplex and others:
#Assumes most recent at the top

Xt = embedded(1,:)
#pointZ = repmat((embedded(1,:)),(n-1),1)
#Need Yi:
Yi = embedded(1:Tp:n-Tp,:);
#And Xi:
Xi = embedded(Tp+1:Tp:n,:);
#vector of distances between latest known point and every other point in library
#i.e Work out (||Xi-Xt||)
far = Xi.-Xt;
FAR = sqrt(sum(far.^2,2));

#calculate weights
weights = exp(-theta.*FAR./distbar);
#calculate Yhat: yhat = A inverse B Xt
#where A = weight(||xi-Xt||)Xi and B = weight(||Xi-Xt||)Yi
#Xi is each point in the library, Xt is the predictee (point) and Yi is where
#Xi ended up at.
m=Xi\Yi
foo=m.*Xt


B = weights.*Yi;
A = weights.*Xi;
Yhat = (A\B).*Xt;
Yhat = sum(Yhat);

#now get the actual true value for the point from the data:
tru = data(1:(latest-1),:);
#always rescale so Xt(0) = 1
tru = tru.+(1-(x(m,:)));


#diagnostics: what happens when theta gets larger?
Yhatvector = 1
iter = 10
for i=1:iter+1
    theta = i-1
    weights = exp(-theta.*FAR./distbar);

B = weights.*Yi;
A = weights.*Xi;
Yhat = (A\B).*Xt;
Yhat = sum(Yhat);
Yhatvector = [Yhatvector;Yhat(1,1)];
endfor

axis = (0:iter)';

#Difference from actual point Yt: 
Yt = tru(length(tru));

closeness = (Yhatvector(2:length(Yhatvector)).-Yt);

#find the value of theta that gives best estimate, and how close it is:
closeness2 = closeness.^2;
[closest2, closetheta] = min(closeness2);
closest=sqrt(closest2)
closetheta = closetheta-1
#use this value of theta:
theta = closetheta
weights = exp(-closetheta.*FAR./distbar);
B = weights.*Yi;
A = weights.*Xi;
Yhat = (A\B).*Xt;
Yhat = sum(Yhat);

#Shows that for logistic map theta=0 is best
#can try other data: 

#plot(axis, closeness, 'x')

plot3 (Xi(:,1), Xi (:,2), weights, 'p');
hold on;
plot3 (Xt(:,1), Xt (:,2), 0, 'o')

##problem: when this is plotted for different values of x the points are displaced
##by the adjustments to start at 1


##to use more data really need to put it into a function first...	



