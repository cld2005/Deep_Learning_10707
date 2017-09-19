function filter_plot(ann,filters)
%mat2gray
for i=0:filters-1
    minv=min(ann.weights{2}(i+1,:));
    maxv=max(ann.weights{2}(i+1,:));
    norm=uint8((ann.weights{2}(i+1,:)-minv)*255/(maxv-minv));
    subplot(10,ceil(filters/10),i+1),imshow(uint8(255*mat2gray(reshape(norm,[28,28]))));
end