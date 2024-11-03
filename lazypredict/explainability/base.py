    def _generate_explainability(self, model: Any, X: Any, y: Any, model_name: str):
        """
        Generates explainability reports using SHAP.
        """
        if self.explainability:
            try:
                # Convert data to pandas DataFrame if necessary
                if self.backend.name != 'pandas':
                    X_explain = X.to_pandas()
                else:
                    X_explain = X

                shap_explainer = ShapExplainer(model, X_explain, use_gpu=self.use_gpu)
                shap_values = shap_explainer.compute_shap_values()
                shap_explainer.plot_shap_summary(shap_values, model_name)
                self.logger.info(f"Generated explainability for {model_name}.")
            except Exception as e:
                self.logger.exception(f"Failed to generate explainability for {model_name}: {e}")
