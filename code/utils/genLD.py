import numpy as np
import math

'''
function ld = genAgeLd(label, sigma, loss)
% genernate label distribution
label_set = 1:4;
switch loss
    case 'klloss'
        ld_num = length(label_set);
        
        dif_age =  bsxfun(@minus,label_set',repmat(label,ld_num,1));
        
        ld = 1./repmat(sqrt(2*pi)*sigma,ld_num,1).*exp(-(dif_age).^2./repmat(2*sigma.^2,ld_num,1));
        
        ld = bsxfun(@times, ld, 1./sum(ld));
    case 'smloss'
        ld = round(label);
    case {'l1', 'l2'}
        ld = 2/84.*(label-1) -1;
end
'''

# generate label distribution
def genLD(label, sigma, loss, class_num):
    label_set = np.array(range(class_num))
    if loss == 'klloss':
        ld_num = len(label_set)
        dif_age = np.tile(label_set.reshape(ld_num, 1), (1, len(label))) - np.tile(label, (ld_num, 1))
        ld = 1.0 / np.tile(np.sqrt(2.0 * np.pi) * sigma, (ld_num, 1)) * np.exp(-1.0 * np.power(dif_age, 2) / np.tile(2.0 * np.power(sigma, 2), (ld_num, 1)))
        ld = ld / np.sum(ld, 0)

        return ld.transpose()


# label = np.array([1, 3, 3, 3])
# sigma = np.array([1, 2.9, 3.0, 4.0])
# loss = 'klloss'
# print(genLD(label, sigma, loss, 4))
#
