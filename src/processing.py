
class Processor():

    def format_context(self, dataframe, location):
        sample = dataframe.loc[location]
        query = sample.query
        title = sample.product_title
        description = f"""{sample.product_description}\n
                        brand: {sample.product_brand}\n
                        color: {sample.product_color}"""
        label = sample.esci_label
        return query, title, description, label

