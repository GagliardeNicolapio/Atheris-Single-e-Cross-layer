import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class Utils {
    public static Instances getInstances(String path) throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(path);
        Instances instances = dataSource.getDataSet();
        instances.setClassIndex(instances.numAttributes()-1);
        return instances;
    }


}
