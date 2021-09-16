
public class TestaSacaNegativo {
	
	public static void main(String[] args) {
		Conta conta = new Conta();
		conta.deposita(100);
		double valorSacado = 101;
		System.out.println(conta.saca(valorSacado));
		
		System.out.println(conta.getSaldo());
	}

}
