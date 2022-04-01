import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;

public class SubsetEval {
    public static void main(String[] args) throws Exception {
        Instances instances = Utils.getInstances("./dataset/dataset.arff");

        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
        BestFirst bestFirst = new BestFirst();

        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(cfsSubsetEval);
        attributeSelection.setSearch(bestFirst);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }
}
