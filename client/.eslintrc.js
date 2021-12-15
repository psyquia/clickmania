module.exports = {
	env: {
		browser: true,
		es2021: true,
	},
	extends: ['plugin:react/recommended', 'airbnb', 'prettier'],
	parserOptions: {
		ecmaFeatures: {
			jsx: true,
		},
		ecmaVersion: 12,
		sourceType: 'module',
	},
	plugins: ['react', 'prettier', 'only-warn'],
	rules: {
		'no-use-before-define': 'off',
		'react/jsx-filename-extension': [2, { extensions: ['.js', '.jsx', '.ts', '.tsx'] }],
		'react/prop-types': 0,
		'react/jsx-props-no-spreading': 'off',
		'import/no-unresolved': 'off',
		'no-console': 'off',
		'no-underscore-dangle': 'off',
		'import/extensions': 'off',
		'spaced-comment': 'off',
		'react/button-has-type': 'off',
		'jsx-a11y/click-events-have-key-events': 'off',
		'jsx-a11y/no-static-element-interactions': 'off',
		'react/no-unescaped-entities': 'off',
		'jsx-a11y/label-has-associated-control': [
			'error',
			{
				required: {
					some: ['nesting', 'id'],
				},
			},
		],
		camelcase: 'off',
		'react/self-closing-comp': 'off',
	},
};
