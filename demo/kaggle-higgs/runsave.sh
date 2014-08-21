mkdir work/$ID && python -u higgs-nfold.py >& work/$ID/higgs.model.log && cp higgs-nfold.py work/$ID && mv higgs.model.* work/$ID
