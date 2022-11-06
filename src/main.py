import model, preparedata, train, test

print('Building Model...')
dlmodel = model.build()

print('Training Model...')
train.train(model=dlmodel)

print('Done')
print('Testing Model...')
test.test()
