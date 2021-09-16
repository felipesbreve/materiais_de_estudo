
public class TestaBanco {
	
	public static void main(String[] args) {
		Cliente felipe = new Cliente();
		felipe.nome = "Felipe Breve";
		felipe.cpf = "123.456.789-10";
		felipe.profissao = "estudante";
		
		Conta minhaConta = new Conta();
		minhaConta.deposita(100);
		
		// Associa a conta com o cliente
		minhaConta.titular = felipe;
		System.out.println(minhaConta.titular.nome);
		System.out.println(minhaConta.titular);
		System.out.println(minhaConta);
	}

}
