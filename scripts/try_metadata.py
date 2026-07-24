from acqstore.acq_image.metadata import ExperimentMetadata, generate_schema_docs

if __name__ == "__main__":
    em = ExperimentMetadata()
    # print(em)
    
    
    # patch = {
    #         'species': 'mouse',
    #         'depth': 1.5,
    #         'branch_order': 2,
    #         'note': 'test note',
    #         # 'xxx_bad_xxx': 'test bad',
    #     }
    # em.update_values(patch=patch)
    # print(em)

    # _schema = em.get_schema()
    # print(_schema)

    # works, needs `uv pip install tabulate`
    
    # either this:
    # print(generate_schema_docs(ExperimentMetadata().get_schema()))
    
    # or this:
    from acqstore.acq_image.metadata import EXPERIMENT_METADATA_SCHEMA
    print(generate_schema_docs(EXPERIMENT_METADATA_SCHEMA))