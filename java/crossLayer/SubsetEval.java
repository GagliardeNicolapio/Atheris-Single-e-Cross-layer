package crossLayer;

import atherisUtils.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;


public class SubsetEval {
    public static void main(String[] args) throws Exception {
        //crossLayer.SubsetEval for cross-layer
        Instances instances = Utils.getInstances("./dataset/datasetDataCleaningScalingFinal.arff");

        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        BestFirst bestFirst = new BestFirst();

        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(cfsSubsetEval);
        attributeSelection.setSearch(bestFirst);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/datasetSubsetEval.csv",instancesEval);

        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }
}
