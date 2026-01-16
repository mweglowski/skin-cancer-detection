# Data analysis
* Dataset consists of:
    * `400666` negatives
    * `393` positives

* Patients count: `1024`

    ![](../../images/eda/distribution_of_counts_across_patients_hist.jpg)
    ![](../../images/eda/distribution_of_counts_across_patients_box.jpg)

* Patients with malignant lesions count: `259`

    ![](../../images/eda/patients_with_most_malignant.jpg)

* Sex

    ![](../../images/eda/counts_of_patients_per_sex.jpg)

* Approximate age

    ![](../../images/eda/distribution_of_ages.jpg)

* Anatomic sites

    ![](../../images/eda/counts_per_anatomic_sites.jpg)

* Attribution

    ![](../../images/eda/counts_per_attribution.jpg)

* Distribution of image shapes. We have randomly sampled 10000 images.

    ```txt
    Number of unique shapes: 82

    Min height: 55
    Max height: 241

    Min width: 55
    Max width: 241
    ```

    ![](../../images/eda/distribution_of_shapes.jpg)