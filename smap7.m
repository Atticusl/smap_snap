##S mapping
#will need to make into a function: I'll do this later. Now needs statistics package. 
#function smap = map(x,d,Tp,theta,y)
load 'logistic.dat' #load up and pre-process a timeseries of data from the logisic map to use as an example
#as it is chaotic.
x=flipud(logistic); #flip so that most recent at top, so not going backwards.
x=logistic (20:50,:);
#may need to then one minus the data if doing
#one step projection, or Tp minus if doing Tp steps.
m=length(x); #define m and d, theta and Tp (last 3 will be in function when done)
d=2;
Tp=1
theta=2
#always rescale so Xt(0) = 1
x = x.+(1-(x(m,:)))
#x=(x(1:m-1)).-(x(2:m)) # '1 minusing' the data reduced length by 1 (or Tp)
#x=(x(1:m-Tp)).-(x(Tp+1:m))
#redefine m: now x i Tp shorter.
m = length(x);

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
embedded
for i= 1:(d+1);
col= x([i:(m-(d-(i-1)))],1);
embedded= [embedded, col];
endfor
# this embedded has the original column of 1's, so chop them off:
embedded= embedded(:,2:(d+2))


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
Yi = embedded(2:Tp:n-Tp,:);
#And Xi:
Xi = embedded(Tp+2:Tp:n,:);
#vector of distances between latest known point and every other point in library
#i.e Work out (||Xi-Xt||)
far = Xi.-Xt;
FAR = sqrt(sum(far.^2,2))

#calculate weights
weights = exp(-theta.*FAR./distbar)
#calculate Yhat: yhat = A inverse B Xt
#where A = weight(||xi-Xt||)Xi and B = weight(||Xi-Xt||)Yi
#Xi is each point in the library, Xt is the predictee (point) and Yi is where
#Xi ended up at.
B = weights.*Yi
A = weights.*Xi
Yhat = (A\B).*Xt
Yhat = sum(Yhat)

#now get the actual true value for the point from the data:
tru = logistic(1:20,:);
#always rescale so Xt(0) = 1
tru = tru.+(1-(x(m,:)))

#diagnostics: what happens when theta gets larger?
Yhatvector = 1
iter = 8
for i=1:iter
    theta = i-1
    weights = exp(-theta.*FAR./distbar);

B = weights.*Yi;
A = weights.*Xi;
Yhat = (A\B).*Xt;
Yhat = sum(Yhat);
Yhatvector = [Yhatvector;Yhat(1,1)];
endfor

axis = (1:iter)'

#Difference from actual point: 
Yt = tru(length(tru))

closeness = (Yhatvector(2:length(Yhatvector)).-Yt)

plot (axis, closeness, 'o')
#shows that most skillful at theta=3
#gets within 0.4, which isn't great.

##to use more data really need to put it into a function first...	

