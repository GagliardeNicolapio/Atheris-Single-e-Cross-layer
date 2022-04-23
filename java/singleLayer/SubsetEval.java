package singleLayer;

import atherisUtils.Utils;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class SubsetEval {
    public static void main(String[] args) throws Exception {
        final String PATH = "./dataset/datasetDataCleaningScalingFinal.arff";
        networkSubsetEval(Utils.networkFeatures(PATH));
        applicationSubsetEval(Utils.applicationFeatures(PATH));
    }

    private static void networkSubsetEval(Instances instances) throws Exception {
        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        BestFirst bestFirst = new BestFirst();

        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(cfsSubsetEval);
        attributeSelection.setSearch(bestFirst);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/subsetEvalDatasetSingleLayerNetwork.csv",instancesEval);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }

    private static void applicationSubsetEval(Instances instances) throws Exception {
        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        BestFirst bestFirst = new BestFirst();

        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(cfsSubsetEval);
        attributeSelection.setSearch(bestFirst);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/subsetEvalDatasetSingleLayerApplication.csv",instancesEval);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }

}
