function filter_plot(ann,filters)

for i=0:filters-1
    subplot(10,ceil(filters/10),i+1),imshow(mat2gray(reshape(ann.weights{2}(i+1,:),[28,28])))
end